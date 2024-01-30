// Package object_tracker implements an object tracker as a Viam vision service
package object_tracker

import (
	"context"
	"fmt"
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
	"image"
)

// ModelName is the name of the model
const (
	ModelName           = "object-tracker"
	MinConfidence       = 0.2
	DefaultMaxFrequency = 10
)

// Here is where we define your new model's colon-delimited-triplet (viam:vision:object-tracker)
var (
	Model            = resource.NewModel("viam", "vision", ModelName)
	errUnimplemented = errors.New("unimplemented")
)

func init() {
	resource.RegisterService(vision.API, Model, resource.Registration[vision.Service, *Config]{
		Constructor: newTracker,
	})
}

func newTracker(ctx context.Context, deps resource.Dependencies, conf resource.Config, logger logging.Logger) (vision.Service, error) {
	t := &myTracker{
		Named:        conf.ResourceName().AsNamed(),
		logger:       logger,
		classCounter: make(map[string]int),
		tracks:       make(map[string][]objdet.Detection),
	}

	if err := t.Reconfigure(ctx, deps, conf); err != nil {
		return nil, err
	}

	// Default value for frequency = 10Hz
	if t.frequency == 0 {
		t.frequency = DefaultMaxFrequency
	}

	// Populate the first set of 2 detections to start us off
	stream, err := t.cam.Stream(ctx, nil)
	if err != nil {
		return nil, err
	}
	for i := 0; i < 2; i++ {
		img, _, err := stream.Next(ctx)
		if err != nil {
			return nil, err
		}
		detections, err := t.detector.Detections(ctx, img, nil)
		if err != nil {
			return nil, err
		}
		t.oldDetections[i] = detections
	}

	filteredOld := FilterDetections(t.chosenLabels, t.oldDetections[0])
	filteredNew := FilterDetections(t.chosenLabels, t.oldDetections[1])

	// Rename from scratch
	// The strategy is to rename the (n-1) detections and then match the (n) dets to those
	// When a (n) det matches a (n-1) det, it copies its label
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
	renamedNew := t.RenameFromMatches(matches, renamedOld, filteredNew)
	t.oldDetections[0] = renamedOld
	t.oldDetections[1] = renamedNew

	return t, nil
}

// Config contains names for necessary resources (camera and vision service)
type Config struct {
	CameraName   string             `json:"camera_name"`
	DetectorName string             `json:"detector_name"`
	ChosenLabels map[string]float64 `json:"chosen_labels"`
	MaxFrequency float64            `json:"max_frequency_hz"`
}

// Validate validates the config and returns implicit dependencies,
// this Validate checks if the camera and detector(vision svc) exist for the module's vision model.
func (cfg *Config) Validate(path string) ([]string, error) {
	// check if the attribute fields for the right and left motors are non-empty
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

type myTracker struct {
	resource.Named
	logger        logging.Logger
	cam           camera.Camera
	camName       string
	detector      vision.Service
	oldDetections [2][]objdet.Detection
	frequency     float64
	chosenLabels  map[string]float64
	classCounter  map[string]int
	tracks        map[string][]objdet.Detection
}

// Reconfigure reconfigures with new settings.
func (t *myTracker) Reconfigure(ctx context.Context, deps resource.Dependencies, conf resource.Config) error {
	t.cam = nil
	t.detector = nil

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
	stream, err := t.cam.Stream(ctx, nil)
	if err != nil {
		return nil, err
	}
	img, _, err := stream.Next(ctx)
	if err != nil {
		return nil, err
	}

	return t.Detections(ctx, img, nil)

}

func (t *myTracker) Detections(ctx context.Context, img image.Image, extra map[string]interface{}) ([]objdet.Detection, error) {
	start := time.Now()

	// Start by grabbing the old detections. They're filtered and renamed already.
	namedOld := t.oldDetections[1]

	// Take fresh detections
	detections, err := t.detector.Detections(ctx, img, nil)
	if err != nil {
		return nil, err
	}
	filteredNew := FilterDetections(t.chosenLabels, detections)

	// Build and solve cost matrix via Munkres' method
	matchMtx := t.BuildMatchingMatrix(namedOld, filteredNew)
	HA, err := hg.NewHungarianAlgorithm(matchMtx)
	if err != nil {
		return nil, err
	}
	out := HA.Execute()

	// Rename new detections the same as old temporal matches
	renamedNew := t.RenameFromMatches(out, namedOld, filteredNew)

	// Then we need to update. The new is now the old yada yada
	// Add Kalman filter stuff here eventually
	t.oldDetections[0] = namedOld
	t.oldDetections[1] = renamedNew

	done := time.Now()
	took := done.Sub(start)
	waitFor := time.Duration((1/t.frequency)*float64(time.Second)) - took
	if waitFor > time.Microsecond {
		time.Sleep(waitFor)
	}

	// Just return the underlying detections
	return renamedNew, nil
}

func (t *myTracker) ClassificationsFromCamera(
	ctx context.Context,
	cameraName string,
	n int,
	extra map[string]interface{},
) (classification.Classifications, error) {
	return nil, errUnimplemented
}

func (t *myTracker) Classifications(ctx context.Context, img image.Image,
	n int, extra map[string]interface{},
) (classification.Classifications, error) {
	return nil, errUnimplemented
}

func (t *myTracker) GetObjectPointClouds(
	ctx context.Context,
	cameraName string,
	extra map[string]interface{},
) ([]*vis.Object, error) {
	return nil, errUnimplemented
}

func (t *myTracker) Close(ctx context.Context) error {
	return nil
}

// DoCommand simply echos whatever was sent.
func (t *myTracker) DoCommand(ctx context.Context, cmd map[string]interface{}) (map[string]interface{}, error) {
	return cmd, nil
}
