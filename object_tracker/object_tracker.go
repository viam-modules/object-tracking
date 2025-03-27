// Package object_tracker implements an object tracker as a Viam vision service
package object_tracker

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"go.viam.com/rdk/gostream"
	"go.viam.com/rdk/vision/viscapture"

	"image"

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
)

// ModelName is the name of the model
const (
	ModelName              = "object-tracker"
	NewObjectDetectedLabel = "new-object-detected"
)

var (
	// Here is where we define your new model's colon-delimited-triplet
	Model                      = resource.NewModel("viam", "vision", ModelName)
	errUnimplemented           = errors.New("unimplemented")
	DefaultMinTrackPersistence = 3
	DefaultMinConfidence       = 0.2
	DefaultMaxFrequency        = 10.0
	DefaultTriggerCoolDown     = 5.0
	DefaultBufferSize          = 30
)

type allObjects struct {
	mutex   sync.RWMutex
	objects []trackedObject
}

type currentDetections struct {
	mutex      sync.RWMutex
	detections []*track
}

func init() {
	resource.RegisterService(vision.API, Model, resource.Registration[vision.Service, *Config]{
		Constructor: newTracker,
	})
}

type myTracker struct {
	resource.Named
	logger        logging.Logger
	cancelFunc    context.CancelFunc
	cancelContext context.Context

	triggerCancelFunc context.CancelFunc
	triggerContext    context.Context

	activeBackgroundWorkers sync.WaitGroup
	lastDetections          []*track
	currDetections          currentDetections
	currImg                 atomic.Pointer[image.Image]
	lostDetectionsBuffer    *tracksBuffer

	allFreshObjects allObjects

	newInstance atomic.Bool
	coolDown    float64
	properties  vision.Properties

	cam                 camera.Camera
	camName             string
	detector            vision.Service
	frequency           float64
	minConfidence       float64
	chosenLabels        map[string]float64
	classCounter        map[string]int
	tracks              map[string][]*track
	timeStats           []time.Duration
	minTrackPersistence int
}

func newTracker(ctx context.Context, deps resource.Dependencies, conf resource.Config, logger logging.Logger) (vision.Service, error) {
	t := &myTracker{
		Named:        conf.ResourceName().AsNamed(),
		logger:       logger,
		classCounter: make(map[string]int),
		tracks:       make(map[string][]*track),
		properties: vision.Properties{
			ClassificationSupported: true,
			DetectionSupported:      true,
			ObjectPCDsSupported:     false,
		},
		allFreshObjects: allObjects{
			objects: []trackedObject{},
		},
		currDetections: currentDetections{},
	}

	if err := t.Reconfigure(ctx, deps, conf); err != nil {
		return nil, err
	}

	//Default value for persistence
	if t.minTrackPersistence == 0 {
		t.minTrackPersistence = DefaultMinTrackPersistence
	}
	// Default value for frequency = 10Hz
	if t.frequency == 0 {
		t.frequency = DefaultMaxFrequency
	}

	cancelableCtx, cancel := context.WithCancel(context.Background())
	t.cancelFunc = cancel
	t.cancelContext = cancelableCtx

	// Do the first pass to populate the first set of 2 detections.
	starterDets := make([][]*track, 2)
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
		filteredDets := FilterDetections(t.chosenLabels, detections, t.minConfidence)
		tracks := newTracks(filteredDets, t.minTrackPersistence)
		starterDets[i] = tracks
	}
	filteredOld := starterDets[0]
	filteredNew := starterDets[1]
	// Rename (from scratch)
	renamedOld := make([]*track, 0, len(filteredOld))
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
	var lostDetections []*track
	for idx := range matches {
		if matches[idx] == -1 {
			// if lost detection is not stable, discard it
			if renamedOld[idx].isStable() {
				lostDetections = append(lostDetections, renamedOld[idx])
			}
		}
	}
	t.lostDetectionsBuffer.AppendDets(lostDetections)

	// Rename from temporal matches. New det copies old det's label
	renamedNew, newlyStable, _ := t.RenameFromMatches(matches, matchMtx, renamedOld, filteredNew)
	if len(newlyStable) > 0 {
		t.trigger()
	}
	renamedNew = append(renamedNew, newlyStable...)
	t.lastDetections = renamedNew
	t.currDetections.mutex.Lock()
	t.currDetections.detections = renamedNew
	t.currDetections.mutex.Unlock()

	t.activeBackgroundWorkers.Add(1)
	viamutils.ManagedGo(func() {
		t.run(stream, t.cancelContext)
	}, func() {
		t.cancelFunc()
		viamutils.UncheckedError(stream.Close(t.cancelContext))
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
			// Take fresh detections from fresh image
			img, _, err := stream.Next(cancelableCtx)
			if err != nil {
				t.logger.Errorf("can't get image. got err: %s", err)
				continue
			}
			if img == nil {
				t.logger.Errorf("got nil image")
				continue
			}
			detections, err := t.detector.Detections(cancelableCtx, img, nil)
			if err != nil {
				t.logger.Errorf("can't get detections. got err: %s", err)
				continue
			}
			filteredDets := FilterDetections(t.chosenLabels, detections, t.minConfidence)
			// all new tracks get a fresh persistence counter
			filteredNew := newTracks(filteredDets, t.minTrackPersistence)

			// Store oldDetection and lost detections in allDetections
			allDetections := t.lastDetections
			for _, dets := range t.lostDetectionsBuffer.detections {
				allDetections = append(allDetections, dets...)
			}
			// Build and solve cost matrix via Munkres' method
			matchMtx := t.BuildMatchingMatrix(allDetections, filteredNew)
			HA, _ := hg.NewHungarianAlgorithm(matchMtx)
			matches := HA.Execute()
			// Store the lost detections in the buffer, drop lost detections
			// if they were not considered stable
			var lostDetections []*track
			for idx := range t.lastDetections {
				if matches[idx] == -1 {
					if t.lastDetections[idx].isStable() {
						lostDetections = append(lostDetections, t.lastDetections[idx])
					} else {
						// drop lost detections from track list as well
						countLabel := getTrackingLabel(t.lastDetections[idx])
						delete(t.tracks, countLabel)
					}
				}
			}
			t.lostDetectionsBuffer.AppendDets(lostDetections)
			// Returns a new set of detections, from matching allDetections with the filteredNew
			// All three outputs must be summed together to get the full set of new detections
			renamedNew, newlyStable, freshDets := t.RenameFromMatches(matches, matchMtx, allDetections, filteredNew)
			if len(newlyStable) > 0 {
				//trigger classification and schedule "untrigger"
				t.trigger()

				// add the detections to the logs
				t.allFreshObjects.mutex.Lock()
				for _, det := range newlyStable {
					to, err := newTrackedObjectFromLabel(det.Det.Label())
					if err != nil {
						t.logger.Error(err)
					}
					t.allFreshObjects.objects = append(t.allFreshObjects.objects, to)
				}
				t.allFreshObjects.mutex.Unlock()
			}
			renamedNew = append(renamedNew, newlyStable...)
			renamedNew = append(renamedNew, freshDets...)
			t.lastDetections = renamedNew
			t.currDetections.mutex.Lock()
			t.currDetections.detections = renamedNew
			t.currDetections.mutex.Unlock()
			t.currImg.Store(&img)

			took := time.Since(start)
			t.timeStats = append(t.timeStats, took)
			waitFor := time.Duration((1/t.frequency)*float64(time.Second)) - took
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

func (t *myTracker) trigger() {
	if t.triggerCancelFunc != nil {
		t.triggerCancelFunc()
	}
	triggerContext, triggerCancelFunc := context.WithCancel(t.cancelContext)
	t.triggerContext = triggerContext
	t.triggerCancelFunc = triggerCancelFunc

	t.newInstance.Store(true)
	t.activeBackgroundWorkers.Add(1)

	viamutils.ManagedGo(
		func() {
			coolDownTimer := time.After(time.Duration(t.coolDown * float64(time.Second)))
			select {
			case <-coolDownTimer:
				t.newInstance.Store(false)
				return
			case <-t.triggerContext.Done():
				return
			}
		},
		func() {
			t.activeBackgroundWorkers.Done()
		})
}

// Config contains names for necessary resources (camera and vision service)
type Config struct {
	CameraName          string             `json:"camera_name"`
	DetectorName        string             `json:"detector_name"`
	ChosenLabels        map[string]float64 `json:"chosen_labels"`
	MaxFrequency        float64            `json:"max_frequency_hz"`
	MinConfidence       *float64           `json:"min_confidence,omitempty"`
	TriggerCoolDown     *float64           `json:"trigger_cool_down_s,omitempty"`
	BufferSize          int                `json:"buffer_size,omitempty"`
	MinTrackPersistence int                `json:"min_track_persistence"`
}

// Validate validates the config and returns implicit dependencies,
// this Validate checks if the camera and detector(vision svc) exist for the module's vision model.
func (cfg *Config) Validate(path string) ([]string, error) {
	if cfg.MinTrackPersistence < 0 {
		return nil, errors.New("attribute min_track_persistence cannot be less than 0")
	}
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

	t.minTrackPersistence = trackerConfig.MinTrackPersistence

	//config buffer size
	if trackerConfig.BufferSize > 0 {
		if trackerConfig.BufferSize > 256 {
			return errors.New("buffer size must be between 1 and 256")
		}
		t.lostDetectionsBuffer = newTracksBuffer(trackerConfig.BufferSize)
	} else {
		t.lostDetectionsBuffer = newTracksBuffer(DefaultBufferSize)
	}

	//config trigger cool down
	if trackerConfig.TriggerCoolDown != nil {
		if *trackerConfig.TriggerCoolDown < 0 {
			return errors.New("trigger_cool_down_s is a duration given in seconds and should be above 0.")
		}
		t.coolDown = *trackerConfig.TriggerCoolDown
	} else {
		t.coolDown = DefaultTriggerCoolDown
	}

	//config min confidence
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
	if cameraName != "" && cameraName != t.camName {
		return nil, errors.Errorf("Camera name given to method, %v is not the same as configured camera %v", cameraName, t.camName)
	}
	select {
	case <-t.cancelContext.Done():
		return nil, t.cancelContext.Err()
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
		t.currDetections.mutex.RLock()
		dets := getStableDetections(t.currDetections.detections)
		t.currDetections.mutex.RUnlock()
		return dets, nil
	}
}

func (t *myTracker) Detections(ctx context.Context, img image.Image, extra map[string]interface{}) ([]objdet.Detection, error) {
	select {
	case <-t.cancelContext.Done():
		return nil, t.cancelContext.Err()
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
		t.currDetections.mutex.RLock()
		dets := getStableDetections(t.currDetections.detections)
		t.currDetections.mutex.RUnlock()
		return dets, nil
	}
}

func (t *myTracker) ClassificationsFromCamera(
	ctx context.Context,
	cameraName string,
	n int,
	extra map[string]interface{},
) (classification.Classifications, error) {
	if cameraName != "" && cameraName != t.camName {
		return nil, errors.Errorf("Camera name given to method, %v is not the same as configured camera %v", cameraName, t.camName)
	}
	if newInstance := t.newInstance.Load(); newInstance {
		return []classification.Classification{classification.NewClassification(1, NewObjectDetectedLabel)}, nil
	} else {
		return []classification.Classification{}, nil
	}
}

func (t *myTracker) Classifications(ctx context.Context, img image.Image,
	n int, extra map[string]interface{},
) (classification.Classifications, error) {
	if newInstance := t.newInstance.Load(); newInstance {
		return []classification.Classification{classification.NewClassification(1, NewObjectDetectedLabel)}, nil
	} else {
		return []classification.Classification{}, nil
	}
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
	var classifications []classification.Classification
	var img image.Image
	select {
	case <-t.cancelContext.Done():
		return viscapture.VisCapture{}, t.cancelContext.Err()
	case <-ctx.Done():
		return viscapture.VisCapture{}, ctx.Err()
	default:
		if opt.ReturnImage {
			if cameraName != "" && cameraName != t.camName {
				return viscapture.VisCapture{}, errors.Errorf("Camera name given to method, %v is not the same as configured camera %v", cameraName, t.camName)
			}
			img = *t.currImg.Load()
		}
		if opt.ReturnDetections {
			t.currDetections.mutex.RLock()
			detections = getStableDetections(t.currDetections.detections)
			t.currDetections.mutex.RUnlock()
		}
		if opt.ReturnClassifications {
			if newInstance := t.newInstance.Load(); newInstance {
				classifications = []classification.Classification{classification.NewClassification(1, NewObjectDetectedLabel)}
			} else {
				classifications = []classification.Classification{}
			}
		}
	}
	return viscapture.VisCapture{Image: img, Detections: detections, Classifications: classifications}, nil
}

func (t *myTracker) Close(ctx context.Context) error {
	t.cancelFunc()
	t.activeBackgroundWorkers.Wait()
	return nil
}

type benchmark struct {
	Slowest      float64
	Fastest      float64
	Average      float64
	NumberOfRuns int
}

// DoCommand will return the slowest, fastest, and average time of the tracking module
func (t *myTracker) DoCommand(ctx context.Context, cmd map[string]interface{}) (map[string]interface{}, error) {
	// average, fastest, and slowest time (and n)
	out := make(map[string]interface{})
	if cmd["benchmark"] != nil {
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
		out["benchmark"] = benchmark{
			Slowest:      float64(tmax),
			Fastest:      float64(tmin),
			Average:      float64(mean),
			NumberOfRuns: int(n),
		}
	}
	if cmd["logs"] != nil {
		t.allFreshObjects.mutex.RLock()
		out["logs"] = t.allFreshObjects.objects
		t.allFreshObjects.mutex.RUnlock()
	}
	return out, nil
}

type tracksBuffer struct {
	detections [][]*track
	size       int
}

// newTracksBuffer initializes a new fixed-length queue with the specified size.
func newTracksBuffer(size int) *tracksBuffer {
	return &tracksBuffer{
		detections: make([][]*track, 0, size),
		size:       size,
	}
}
func (b *tracksBuffer) AppendDets(newDets []*track) {
	if len(b.detections) == b.size {
		b.detections = b.detections[1:]
	}

	//remove old dets to match new dets only on the most recent detections
	for _, newDet := range newDets {
		countLabel := strings.Join(strings.Split(newDet.Det.Label(), "_")[0:2], "_")
		for i := range b.detections {
			dets := b.detections[i]
			for idx, det := range dets {
				oldCountLabel := strings.Join(strings.Split(det.Det.Label(), "_")[0:2], "_")
				if countLabel == oldCountLabel {
					b.detections[i] = append(dets[:idx], dets[idx+1:]...)
					break
				}
			}
		}
	}

	b.detections = append(b.detections, newDets)
}
