// Package object_tracker implements an object tracker as a Viam vision service
package object_tracker

import (
	objdet "go.viam.com/rdk/vision/objectdetection"
	"strings"
)

// NewAdvancedFilter returns a Detections->Detections filtering method to remove
// detections that do not have a class name in chosenLabels and/or do not have the
// associated minimum confidence. An empty input map will return all detections.
// Input chosenLabels is the map with <"class_name": confidence> key-value pairs.
func NewAdvancedFilter(chosenLabels map[string]float64) objdet.Postprocessor {
	return func(detections []objdet.Detection) []objdet.Detection {
		// If it's empty, return the input.
		if len(chosenLabels) < 1 {
			return detections
		}

		out := make([]objdet.Detection, 0, len(detections))
		for _, d := range detections {
			minConf, ok := chosenLabels[strings.ToLower(d.Label())]
			if ok {
				if d.Score() > minConf {
					out = append(out, d)
				}
			}
		}
		return out
	}
}
