package object_tracker

import (
	"strconv"
	"strings"

	"github.com/pkg/errors"
	objdet "go.viam.com/rdk/vision/objectdetection"
)

// A track stores information about the bounding box as well as its persistence properties
// across frames
type track struct {
	Det              objdet.Detection
	persistenceLimit int
	persistenceCount int
	stable           bool
}

// newTrack turns a bounding box into a new track with a fresh persistence counter
func newTrack(det objdet.Detection, lim int) *track {
	return &track{det, lim, 0, false}
}

// newTracks turns a slice of bounding boxes into a track with a fresh persistence counter
func newTracks(dets []objdet.Detection, lim int) []*track {
	tracks := make([]*track, 0, len(dets))
	for _, d := range dets {
		tracks = append(tracks, newTrack(d, lim))
	}
	return tracks
}

// clone will duplicate all the properties of the track
func (tr *track) clone() *track {
	return &track{
		tr.Det,
		tr.persistenceLimit,
		tr.persistenceCount,
		tr.stable,
	}
}

// isStable returns that the track has persisted long enough to count as stable
func (tr *track) isStable() bool {
	return tr.stable
}

// addPersistence add to the persistence counter
func (tr *track) addPersistence() {
	if tr.stable {
		return
	}
	tr.persistenceCount += 1
	if tr.persistenceCount >= tr.persistenceLimit {
		tr.stable = true
	}
}

// return only the bounding boxes associated with stable tracks
func getStableDetections(tracks []*track) []objdet.Detection {
	dets := make([]objdet.Detection, 0, len(tracks))
	for _, tr := range tracks {
		if tr.stable {
			dets = append(dets, tr.Det)
		}
	}
	return dets
}

// trackedObject is the log info associated with the track that is stable
type trackedObject struct {
	FullLabel string
	Label     string
	Id        int
	Time      string
}

func newTrackedObjectFromLabel(label string) (trackedObject, error) {
	parts := strings.Split(label, "_")
	id, err := strconv.Atoi(parts[1])
	if err != nil {
		return trackedObject{}, errors.Wrapf(err, "unable to parse label %v", label)
	}
	return trackedObject{
		FullLabel: label,
		Label:     parts[0],
		Id:        id,
		Time:      strings.Join(parts[2:], "_"),
	}, nil

}
