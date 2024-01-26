// Package object_tracker implements an object tracker as a Viam vision service
package object_tracker

import (
	"fmt"
	objdet "go.viam.com/rdk/vision/objectdetection"
	"strconv"
	"strings"
	"time"
)

// GetTimestamp will retrieve and format a timestamp to be YYYYMMDD_HHMM
func GetTimestamp() string {
	currTime := time.Now()
	return fmt.Sprintf(currTime.Format("20060201_1504"))
}

// ReplaceLabel replaces the detection with an almost identical detection (new label)
func ReplaceLabel(det objdet.Detection, label string) objdet.Detection {
	return objdet.NewDetection(*det.BoundingBox(), det.Score(), label)
}

// RenameFromMatches takes the output of the Hungarian matching algorithm and
// gives the new detection the same label as the matching old detection.  Any new detections
// found will be given a new name (and class counter will be updated)
func (t *myTracker) RenameFromMatches(matches []int, oldDets, newDets []objdet.Detection) []objdet.Detection {
	// Fill up a map with the indices of newDetections we have
	notUsed := make(map[int]struct{})
	for i, _ := range newDets {
		notUsed[i] = struct{}{}
	}
	fmt.Printf("THE MATCHES HERE ---> %v\n", matches)
	fmt.Printf("The len of newDets--> %v\n", len(newDets))

	for oldIdx, newIdx := range matches {
		if newIdx != -1 {
			newDets[newIdx] = ReplaceLabel(newDets[newIdx], oldDets[oldIdx].Label())
			delete(notUsed, newIdx)
		}
	}
	if len(newDets) > len(matches) {
		for idx := range notUsed {
			newDets[idx] = t.RenameFirstTime(newDets[idx])
		}
	}
	return newDets
}

// RenameFirstTime should activate whenever a new object appears.
// It will start or update a class counter for whichever class.
func (t *myTracker) RenameFirstTime(det objdet.Detection) objdet.Detection {
	baseLabel := strings.ToLower(strings.Split(det.Label(), "_")[0])
	classCount, ok := t.classCounter[baseLabel]
	if !ok {
		t.classCounter[baseLabel] = 0
	} else {
		t.classCounter[baseLabel] = classCount + 1
	}
	label := baseLabel + "_" + strconv.Itoa(t.classCounter[baseLabel]) + "_" + GetTimestamp()
	return objdet.NewDetection(*det.BoundingBox(), det.Score(), label)
}
