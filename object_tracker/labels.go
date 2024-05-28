// Package object_tracker implements an object tracker as a Viam vision service.
// This file contains methods that handle the label (or name) of a detection
// If two detections are output with the same label, they are considered the same object
// Labels are of the format classname_N_YYYYMMDD_HHMM
package object_tracker

import (
	"fmt"
	objdet "go.viam.com/rdk/vision/objectdetection"
	"strconv"
	"strings"
	"time"
)

// GetTimestamp will retrieve and format a timestamp to be YYYYMMDD_HHMMSS
func GetTimestamp() string {
	currTime := time.Now()
	return fmt.Sprintf(currTime.Format("20060102_150405"))
}

// ReplaceLabel replaces the detection with an almost identical detection (new label)
func ReplaceLabel(det objdet.Detection, label string) objdet.Detection {
	return objdet.NewDetection(*det.BoundingBox(), det.Score(), label)
}

// RenameFromMatches takes the output of the Hungarian matching algorithm and
// gives the new detection the same label as the matching old detection.  Any new detections
// found will be given a new name (and class counter will be updated)
// Also return
func (t *myTracker) RenameFromMatches(matches []int, oldDets, newDets []objdet.Detection) ([]objdet.Detection, []objdet.Detection) {
	// Fill up a map with the indices of newDetections we have
	notUsed := make(map[int]struct{})
	for i, _ := range newDets {
		notUsed[i] = struct{}{}
	}
	// Go through valid matches and update name and track
	for oldIdx, newIdx := range matches {
		if newIdx != -1 {
			newDets[newIdx] = ReplaceLabel(newDets[newIdx], oldDets[oldIdx].Label())
			t.UpdateTrack(newDets[newIdx])
			delete(notUsed, newIdx)
		}
	}
	// Go through all NEW things and add them in (name them and start new track)
	var newNewDets []objdet.Detection
	if len(newDets) > len(matches) {
		for idx := range notUsed {
			newDet := t.RenameFirstTime(newDets[idx])
			newDets[idx] = newDet
			newNewDets = append(newNewDets, newDet)
		}
	}
	return newDets, newNewDets
}

// RenameFirstTime should activate whenever a new object appears.
// It will start or update a class counter for whichever class and create a new track.
func (t *myTracker) RenameFirstTime(det objdet.Detection) objdet.Detection {
	baseLabel := strings.ToLower(strings.Split(det.Label(), "_")[0])
	classCount, ok := t.classCounter[baseLabel]
	if !ok {
		t.classCounter[baseLabel] = 0
	} else {
		t.classCounter[baseLabel] = classCount + 1
	}
	countLabel := baseLabel + "_" + strconv.Itoa(t.classCounter[baseLabel])
	label := countLabel + "_" + GetTimestamp()
	out := objdet.NewDetection(*det.BoundingBox(), det.Score(), label)
	t.tracks[countLabel] = []objdet.Detection{out} // Start new track with this one
	return out
}

func (t *myTracker) UpdateTrack(det objdet.Detection) {
	countLabel := strings.Join(strings.Split(det.Label(), "_")[0:2], "_")
	track, ok := t.tracks[countLabel]
	if ok {
		t.tracks[countLabel] = append(track, det)
	}
}
