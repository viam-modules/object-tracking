// Package object_tracker implements an object tracker as a Viam vision service
package object_tracker

import (
	"context"
	"fmt"
	"go.viam.com/rdk/gostream"
	"go.viam.com/rdk/vision/viscapture"
	"sync"
	"sync/atomic"
	"time"

	hg "github.com/charles-haynes/munkres"
	"github.com/pkg/errors"
	"go.viam.com/rdk/components/camera"
	"go.viam.com/rdk/logging"
	"go.viam.com/rdk/resource"
	"go.viam.com/rdk/services/vision"
	vis "go.viam.com/rdk/vision"
	"go.viam.com/rdk/vision/classification"
	objdet "go.viam.com/rdk/vision/objectdetection"
	viamutils "go.viam.com/utils"
	"image"
)

// ModelName is the name of the model
const (
	ModelName = "object-tracker"
)

var (
	// Here is where we define your new model's colon-delimited-triplet (viam:vision:object-tracker)
	Model                = resource.NewModel("viam", "vision", ModelName)
	errUnimplemented     = errors.New("unimplemented")
	DefaultMinConfidence = 0.2
	DefaultMaxFrequency  = 10.0
)

func init() {
	resource.RegisterService(vision.API, Model, resource.Registration[vision.Service, *Config]{
		Constructor: newTracker,
	})
}

type myTracker struct {
	resource.Named
	logger                  logging.Logger
	cancelFunc              context.CancelFunc
	cancelContext           context.Context
	activeBackgroundWorkers sync.WaitGroup
	oldDetections           atomic.Pointer[[2][]objdet.Detection]
	currImg                 atomic.Pointer[image.Image]
	allClass                atomic.Pointer[classification.Classifications]

	channel chan []objdet.Detection

	newInstance atomic.Bool
	coolDown    float64
	properties  vision.Properties

	cam           camera.Camera
	camName       string
	detector      vision.Service
	frequency     float64
	minConfidence float64
	chosenLabels  map[string]float64
	classCounter  map[string]int
	tracks        map[string][]objdet.Detection
	timeStats     []time.Duration
}

func newTracker(ctx context.Context, deps resource.Dependencies, conf resource.Config, logger logging.Logger) (vision.Service, error) {

	t := &myTracker{
		Named:        conf.ResourceName().AsNamed(),
		logger:       logger,
		classCounter: make(map[string]int),
		tracks:       make(map[string][]objdet.Detection),
		properties: vision.Properties{
			ClassificationSupported: false,
			DetectionSupported:      true,
			ObjectPCDsSupported:     false,
		},
		coolDown: 2,
		channel:  make(chan []objdet.Detection, 1024),
	}

	if err := t.Reconfigure(ctx, deps, conf); err != nil {
		return nil, err
	}
	// Default value for frequency = 10Hz
	if t.frequency == 0 {
		t.frequency = DefaultMaxFrequency
	}

	cancelableCtx, cancel := context.WithCancel(context.Background())
	t.cancelFunc = cancel
	t.cancelContext = cancelableCtx

	// Do the first pass to populate the first set of 2 detections.
	starterDets := make([][]objdet.Detection, 2)
	stream, err := t.cam.Stream(t.cancelContext, nil)
	if err != nil {
		return nil, err
	}
	for i := 0; i < 2; i++ {
		img, _, err := stream.Next(t.cancelContext)
		if err != nil {
			return nil, err
		}
		detections, err := t.detector.Detections(ctx, img, nil)
		if err != nil {
			return nil, err
		}
		starterDets[i] = detections
	}
	filteredOld := FilterDetections(t.chosenLabels, starterDets[0], t.minConfidence)
	filteredNew := FilterDetections(t.chosenLabels, starterDets[1], t.minConfidence)
	// Rename (from scratch)
	renamedOld := make([]objdet.Detection, 0, len(filteredOld))
	for _, det := range filteredOld {
		newDet := t.RenameFirstTime(det)
		renamedOld = append(renamedOld, newDet)
	}
	// Build and solve cost matrix via Munkres' method
	matchMtx := t.BuildMatchingMatrix(renamedOld, filteredNew)
	HA, err := hg.NewHungarianAlgorithm(matchMtx)
	if err != nil {
		return nil, err
	}
	matches := HA.Execute()
	// Rename from temporal matches. New det copies old det's label
	renamedNew, _ := t.RenameFromMatches(matches, renamedOld, filteredNew)
	t.oldDetections.Store(&[2][]objdet.Detection{renamedOld, renamedNew})

	t.activeBackgroundWorkers.Add(1)
	viamutils.ManagedGo(func() {
		t.run(stream, t.cancelContext)
	}, func() {
		t.cancelFunc()
		stream.Close(t.cancelContext)
		t.activeBackgroundWorkers.Done()
	})

	return t, nil
}

// run is a (cancelable) infinite loop that takes new detections from the camera and compares them to
// the most recently seen detections. Matching detections are linked via matching labels.
func (t *myTracker) run(stream gostream.VideoStream, cancelableCtx context.Context) {
	for {
		select {
		case <-cancelableCtx.Done():
			return
		default:
			start := time.Now()
			// Load up the old detections
			namedOld := t.oldDetections.Load()[1]

			// Take fresh detections from fresh image
			img, _, err := stream.Next(cancelableCtx)
			if err != nil {
				t.logger.Error(err)
				return
			}
			detections, err := t.detector.Detections(cancelableCtx, img, nil)
			if err != nil {
				t.logger.Error(err)
				return
			}
			filteredNew := FilterDetections(t.chosenLabels, detections, t.minConfidence)

			// Build and solve cost matrix via Munkres' method
			matchMtx := t.BuildMatchingMatrix(namedOld, filteredNew)
			HA, _ := hg.NewHungarianAlgorithm(matchMtx)
			matches := HA.Execute()
			// Rename from temporal matches. New det copies old det's label
			curDets, newDets := t.RenameFromMatches(matches, namedOld, filteredNew)
			if len(newDets) > 0 {
				t.channel <- newDets
			}

			// Store the matched detections and image
			t.oldDetections.Store(&[2][]objdet.Detection{namedOld, curDets})
			t.currImg.Store(&img)

			took := time.Since(start)
			waitFor := time.Duration((1/t.frequency)*float64(time.Second)) - took
			t.timeStats = append(t.timeStats, took)
			if waitFor > time.Microsecond {
				select {
				case <-cancelableCtx.Done():
					return
				case <-time.After(waitFor):
				}
			}
		}
	}
}

// Config contains names for necessary resources (camera and vision service)
type Config struct {
	CameraName    string             `json:"camera_name"`
	DetectorName  string             `json:"detector_name"`
	ChosenLabels  map[string]float64 `json:"chosen_labels"`
	MaxFrequency  float64            `json:"max_frequency_hz"`
	MinConfidence *float64           `json:"min_confidence,omitempty"`
}

// Validate validates the config and returns implicit dependencies,
// this Validate checks if the camera and detector(vision svc) exist for the module's vision model.
func (cfg *Config) Validate(path string) ([]string, error) {
	// this makes them required for the model to successfully build
	if cfg.CameraName == "" {
		return nil, fmt.Errorf(`expected "camera_name" attribute for object tracker %q`, path)
	}
	if cfg.DetectorName == "" {
		return nil, fmt.Errorf(`expected "detector_name" attribute for object tracker %q`, path)
	}

	// Return the resource names so that newTracker can access them as dependencies.
	return []string{cfg.CameraName, cfg.DetectorName}, nil
}

// Reconfigure reconfigures with new settings.
func (t *myTracker) Reconfigure(ctx context.Context, deps resource.Dependencies, conf resource.Config) error {
	var timeList []time.Duration
	t.cam = nil
	t.detector = nil
	t.timeStats = timeList

	// This takes the generic resource.Config passed down from the parent and converts it to the
	// model-specific (aka "native") Config structure defined, above making it easier to directly access attributes.
	trackerConfig, err := resource.NativeConfig[*Config](conf)
	if err != nil {
		return errors.Errorf("Could not assert proper config for %s", ModelName)
	}

	if trackerConfig.MaxFrequency < 0 {
		// if 0, will be set to default later
		return errors.New("frequency(Hz) must be a positive number")
	}
	t.frequency = trackerConfig.MaxFrequency

	if trackerConfig.MinConfidence != nil {
		t.minConfidence = *trackerConfig.MinConfidence
	} else {
		t.minConfidence = DefaultMinConfidence
	}
	if t.minConfidence < 0 || t.minConfidence > 1 {
		return errors.New("minimum thresholding confidence must be between 0.0 and 1.0")
	}

	t.chosenLabels = trackerConfig.ChosenLabels
	t.camName = trackerConfig.CameraName
	t.cam, err = camera.FromDependencies(deps, trackerConfig.CameraName)
	if err != nil {
		return errors.Wrapf(err, "unable to get camera %v for object tracker", trackerConfig.CameraName)
	}
	t.detector, err = vision.FromDependencies(deps, trackerConfig.DetectorName)
	if err != nil {
		return errors.Wrapf(err, "unable to get camera %v for object tracker", trackerConfig.DetectorName)
	}
	return nil
}

func (t *myTracker) DetectionsFromCamera(
	ctx context.Context,
	cameraName string,
	extra map[string]interface{},
) ([]objdet.Detection, error) {
	if cameraName != t.camName {
		return nil, errors.Errorf("Camera name given to method, %v is not the same as configured camera %v", cameraName, t.camName)
	}
	select {
	case <-t.cancelContext.Done():
		return nil, t.cancelContext.Err()
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
		return t.oldDetections.Load()[1], nil
	}
}

func (t *myTracker) Detections(ctx context.Context, img image.Image, extra map[string]interface{}) ([]objdet.Detection, error) {
	select {
	case <-t.cancelContext.Done():
		return nil, t.cancelContext.Err()
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
		return t.oldDetections.Load()[1], nil
	}
}

func (t *myTracker) ClassificationsFromCamera(
	ctx context.Context,
	cameraName string,
	n int,
	extra map[string]interface{},
) (classification.Classifications, error) {
	//var classifications classification.Classifications
	if cameraName != t.camName {
		return nil, errors.Errorf("Camera name given to method, %v is not the same as configured camera %v", cameraName, t.camName)
	}
	//var dets []objdet.Detection
	var res []classification.Classification

	for {
		select {
		case <-t.cancelContext.Done():
			return nil, t.cancelContext.Err()
		case <-ctx.Done():
			return nil, ctx.Err()
		case dets, ok := <-t.channel:
			if !ok {
				// The channel is closed
				t.logger.Error("CHANNEL CLOSED")
				return res, nil
			}
			t.logger.Errorf("GOT DETS : %s", dets)
			for _, det := range dets {
				label := det.Label()
				res = append(res, classification.NewClassification(1, label))
			}
		default:
			return res, nil
		}
	}
	//select {
	//case <-t.cancelContext.Done():
	//	return nil, t.cancelContext.Err()
	//case <-ctx.Done():
	//	return nil, ctx.Err()
	//case dets, ok := <-t.channel:
	//	if !ok {
	//		// The channel is closed
	//		t.logger.Error("CHANNEL CLOSED")
	//		return res, nil
	//	}
	//for i := 0; i < t.maxBufferSize; i++ {
	//	t.logger.Errorf("Start iteration %d", i)
	//	dets = t.consumer.Get() //loops over the buffer
	//	t.logger.Errorf("GOT %d", dets)
	//	if dets == nil {
	//		continue
	//	} else {
	//		for _, det := range dets {
	//			label := det.Label()
	//			res[i] = classification.NewClassification(1, label)
	//		}
	//	}
	//
	//}
	//	t.logger.Error("READING FROM CHANNEL")
	//	for dets := range t.channel {
	//		t.logger.Errorf("GOT DETS : %s", dets)
	//		for _, det := range dets {
	//			label := det.Label()
	//			res = append(res, classification.NewClassification(1, label))
	//		}
	//	}
	//	t.logger.Error("READING FROM CHANNEL")
	//	return res, nil
	//default:
	//	return nil, nil
	//}
}

func (t *myTracker) Classifications(ctx context.Context, img image.Image,
	n int, extra map[string]interface{},
) (classification.Classifications, error) {
	return nil, errUnimplemented
}

func (t *myTracker) GetProperties(ctx context.Context, extra map[string]interface{}) (*vision.Properties, error) {
	return &t.properties, nil
}
func (t *myTracker) GetObjectPointClouds(
	ctx context.Context,
	cameraName string,
	extra map[string]interface{},
) ([]*vis.Object, error) {
	return nil, errUnimplemented
}

func (t *myTracker) CaptureAllFromCamera(
	ctx context.Context,
	cameraName string,
	opt viscapture.CaptureOptions,
	extra map[string]interface{},
) (viscapture.VisCapture, error) {
	var detections []objdet.Detection
	var img image.Image
	select {
	case <-t.cancelContext.Done():
		return viscapture.VisCapture{}, t.cancelContext.Err()
	case <-ctx.Done():
		return viscapture.VisCapture{}, ctx.Err()
	default:
		if opt.ReturnImage {
			if cameraName != t.camName {
				return viscapture.VisCapture{}, errors.Errorf("Camera name given to method, %v is not the same as configured camera %v", cameraName, t.camName)
			}
			img = *t.currImg.Load()
		}
		if opt.ReturnDetections {
			detections = t.oldDetections.Load()[1]
		}
	}
	return viscapture.VisCapture{Image: img, Detections: detections}, nil
}

func (t *myTracker) Close(ctx context.Context) error {
	t.cancelFunc()
	t.activeBackgroundWorkers.Wait()
	return nil
}

// DoCommand will return the slowest, fastest, and average time of the tracking module
func (t *myTracker) DoCommand(ctx context.Context, cmd map[string]interface{}) (map[string]interface{}, error) {
	// average, fastest, and slowest time (and n)
	tmin, tmax := 10*time.Second, 10*time.Nanosecond
	n := int64(len(t.timeStats))
	var sum time.Duration
	for _, tt := range t.timeStats {
		if tt < tmin {
			tmin = tt
		}
		if tt > tmax {
			tmax = tt
		}
		sum += tt
	}
	mean := time.Duration(int64(sum) / n)
	out := map[string]interface{}{
		"slowest":        fmt.Sprintf("%s", tmax),
		"fastest":        fmt.Sprintf("%s", tmin),
		"average":        fmt.Sprintf("%s", mean),
		"number of runs": fmt.Sprintf("%v", n),
	}
	return out, nil
}
