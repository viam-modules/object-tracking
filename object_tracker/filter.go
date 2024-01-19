// Package object_tracker does the object tracking or whateva
package object_tracker

import (
	objdet "go.viam.com/rdk/vision/objectdetection"
	"strings"
)

func NewAdvancedFilter(chosenLabels map[string]float64) objdet.Postprocessor {
	return func(detections []objdet.Detection) []objdet.Detection {
		// If it's empty don't bother. Return the input.
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
		// For each detection look at it. Check if the class is in chosen labels
		// If it is, check that the confidence is above the corresponding shit.
		// If it is, cool. Include it. If at any point it fucks up, remove it.
	}
}
