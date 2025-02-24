// Package object_tracker implements an object tracker as a Viam vision service.
// This file contains methods that handle the label (or name) of a detection
// If two detections are output with the same label, they are considered the same object
// Labels are of the format classname_N_YYYYMMDD_HHMM
package object_tracker

import (
	"image"
	"strconv"
	"strings"
	"time"

	objdet "go.viam.com/rdk/vision/objectdetection"
)

// GetTimestamp will retrieve and format a timestamp to be YYYYMMDD_HHMMSS
func GetTimestamp() string {
	currTime := time.Now()
	return currTime.Format("20060102_150405")
}

// ReplaceLabel replaces the detection with an almost identical detection (new label)
func ReplaceLabel(tr *track, label string) *track {
	det := objdet.NewDetection(*tr.Det.BoundingBox(), tr.Det.Score(), label)
	newTrack := tr.clone()
	newTrack.Det = det
	return newTrack
}

// ReplaceBoundingBox replaces the detection with an almost identical detection (new bounding box)
func ReplaceBoundingBox(tr *track, bb *image.Rectangle) *track {
	det := objdet.NewDetection(*bb, tr.Det.Score(), tr.Det.Label())
	newTrack := tr.clone()
	newTrack.Det = det
	return newTrack
}

// RenameFromMatches takes the output of the Hungarian matching algorithm and
// gives the new detection the same label as the matching old detection.  Any new detections
// found will be given a new name (and cleass counter will be updated)
// Also return freshDets that are the fresh detections that were not matched with any detections in the previous frame.
func (t *myTracker) RenameFromMatches(matches []int, matchinMtx [][]float64, oldDets, newDets []*track) ([]*track, []*track, []*track) {
	// Fill up a map with the indices of newDetections we have
	notUsed := make(map[int]struct{})
	for i, _ := range newDets {
		notUsed[i] = struct{}{}
	}
	// Go through valid matches and update name and track
	updatedTracks := make([]*track, 0)
	newlyStableTracks := make([]*track, 0)
	for oldIdx, newIdx := range matches {
		if newIdx != -1 {
			if matchinMtx[oldIdx][newIdx] != 0 {
				if newIdx >= 0 && newIdx < len(newDets) && oldIdx >= 0 && oldIdx < len(oldDets) {
					// take the old track, clone it, and update their Bounding Box
					// to the new track. Increment its persistence counter.
					updatedTrack, newlyStable := t.UpdateTrack(newDets[newIdx], oldDets[oldIdx])
					if newlyStable {
						newlyStableTracks = append(newlyStableTracks, updatedTrack)
					} else {
						updatedTracks = append(updatedTracks, updatedTrack)
					}
					delete(notUsed, newIdx)
				}
			}
		}
	}
	// Go through all NEW things and add them in (name them and start new track)
	freshTracks := make([]*track, 0)
	for idx := range notUsed {
		newDet := t.RenameFirstTime(newDets[idx])
		newDets[idx] = newDet
		freshTracks = append(freshTracks, newDet)
	}
	return updatedTracks, newlyStableTracks, freshTracks
}

// RenameFirstTime should activate whenever a new object appears.
// It will start or update a class counter for whichever class and create a new track.
func (t *myTracker) RenameFirstTime(det *track) *track {
	baseLabel := strings.ToLower(strings.Split(det.Det.Label(), "_")[0])
	classCount, ok := t.classCounter[baseLabel]
	if !ok {
		t.classCounter[baseLabel] = 0
	} else {
		t.classCounter[baseLabel] = classCount + 1
	}
	countLabel := baseLabel + "_" + strconv.Itoa(t.classCounter[baseLabel])
	label := countLabel + "_" + GetTimestamp()
	out := ReplaceLabel(det, label)
	// start a new track, but it will be tentative, and may be removed if lost
	// before persistence counter reaches "stable"
	t.tracks[countLabel] = []*track{out}
	return out
}

func getTrackingLabel(tr *track) string {
	return strings.Join(strings.Split(tr.Det.Label(), "_")[0:2], "_")
}

// UpdateTrack changes the old bounding box to the new one, updates persistence,
// and also returns if the track became newly stable
func (t *myTracker) UpdateTrack(nextTrack, oldMatchedTrack *track) (*track, bool) {
	wasStable := oldMatchedTrack.isStable()
	newTrack := ReplaceBoundingBox(oldMatchedTrack, nextTrack.Det.BoundingBox())
	newTrack.addPersistence()
	countLabel := getTrackingLabel(newTrack)
	trackSlice, ok := t.tracks[countLabel]
	if ok {
		t.tracks[countLabel] = append(trackSlice, newTrack)
	}
	isNowStable := newTrack.isStable()
	newlyStable := wasStable != isNowStable
	return newTrack, newlyStable
}
