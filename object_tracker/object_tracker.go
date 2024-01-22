// Package object_tracker does the object tracking or whateva
package object_tracker

import (
	"context"
	"fmt"
	"github.com/pkg/errors"
	"go.viam.com/rdk/components/camera"
	"go.viam.com/rdk/gostream"
	"go.viam.com/rdk/logging"
	"go.viam.com/rdk/resource"
	"go.viam.com/rdk/services/vision"
	vis "go.viam.com/rdk/vision"
	"go.viam.com/rdk/vision/classification"
	objdet "go.viam.com/rdk/vision/objectdetection"
	"image"
)

// ModelName is the name of the model
const ModelName = "object-tracker"

// Here is where we define your new model's colon-delimited-triplet (acme:demo:mybase)
// acme = namespace, demo = repo-name, mybase = model name.
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
		Named:  conf.ResourceName().AsNamed(),
		logger: logger,
	}
	// This will set the t.cam and t.detector
	if err := t.Reconfigure(ctx, deps, conf); err != nil {
		return nil, err
	}

	// Populate the first set of 5 detections with images from camera
	stream, err := t.cam.Stream(ctx, nil)
	if err != nil {
		return nil, err
	}
	t.camStream = stream
	for i := 0; i < 5; i++ {
		img, _, err := t.camStream.Next(ctx)
		if err != nil {
			return nil, err
		}
		detections, err := t.detector.Detections(ctx, img, nil)
		if err != nil {
			return nil, err
		}
		t.detections[i] = detections
	}

	return t, nil
}

// Config contains two component (motor) names.
type Config struct {
	CameraName   string             `json:"camera_name"`
	DetectorName string             `json:"detector_name"`
	ChosenLabels map[string]float64 `json:"chosen_labels"`
}

// Validate validates the config and returns implicit dependencies,
// this Validate checks if the left and right motors exist for the module's base model.
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
	logger       logging.Logger
	cam          camera.Camera
	camStream    gostream.VideoStream
	detector     vision.Service
	detections   [5][]objdet.Detection
	chosenLabels map[string]float64
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

	t.chosenLabels = trackerConfig.ChosenLabels // needs some validation but yeah.

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

	// What's crazy is the transfrom camera actually uses Detections not DetsFromCam
	// Need to check cameraName against config and then call Detections but for now...
	return t.detections[3], nil

}

func (t *myTracker) Detections(ctx context.Context, img image.Image, extra map[string]interface{},
) ([]objdet.Detection, error) {

	detections, err := t.detector.Detections(ctx, img, nil)
	if err != nil {
		return nil, err
	}
	outDets := NewAdvancedFilter(t.chosenLabels)(detections)

	// Just return the underlying detections
	return outDets, nil

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

// Close stops motion during shutdown.
func (t *myTracker) Close(ctx context.Context) error {
	return nil
}

// DoCommand simply echos whatever was sent.
func (t *myTracker) DoCommand(ctx context.Context, cmd map[string]interface{}) (map[string]interface{}, error) {
	return cmd, nil
}
