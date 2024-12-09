package object_tracker

import (
	"image"
	"testing"

	hg "github.com/charles-haynes/munkres"
	"go.viam.com/rdk/services/vision"
	objdet "go.viam.com/rdk/vision/objectdetection"
	"go.viam.com/test"
)

const (
	LabelDet0 string = "cat"
	LabelDet1 string = "fish"
)

type FakeDetector struct {
	it  int
	res [][]objdet.Detection
}

func (fd *FakeDetector) fakeDetections() []objdet.Detection {
	fd.it += 1
	return fd.res[fd.it-1]
}

func checkLabel(t *testing.T, value objdet.Detection, target string) {
	test.That(t, value.Label()[:len(target)], test.ShouldEqual, target)
}

func TestTracker(t *testing.T) {
	det0 := objdet.NewDetection(image.Rect(0, 0, 10, 10), 1, LabelDet0)
	det1 := objdet.NewDetection(image.Rect(20, 20, 30, 30), 1, LabelDet1)
	det1_1 := objdet.NewDetection(image.Rect(22, 22, 33, 33), 1, LabelDet1)
	detsT0 := []objdet.Detection{det0, det1}
	detsT1 := []objdet.Detection{det0}
	detsT2 := []objdet.Detection{det1_1}
	detsT3 := []objdet.Detection{det0}

	fd := &FakeDetector{
		res: [][]objdet.Detection{detsT0, detsT1, detsT2, detsT3},
	}

	fakeTracker := &myTracker{
		classCounter: make(map[string]int),
		tracks:       make(map[string][]objdet.Detection),
		properties: vision.Properties{
			ClassificationSupported: true,
			DetectionSupported:      true,
			ObjectPCDsSupported:     false,
		},
		allFreshObjects: allObjects{
			objects: []trackedObject{},
		},
		lostDetectionsBuffer: newDetectionsBuffer(10),
	}

	//initialisation
	filteredOld := fd.fakeDetections() // get cat and fish
	filteredNew := fd.fakeDetections() //get cat
	renamedOld := make([]objdet.Detection, 0, len(filteredOld))
	for _, det := range filteredOld {
		newDet := fakeTracker.RenameFirstTime(det) //create label fish_0 and cat_0
		renamedOld = append(renamedOld, newDet)
	}
	//fakeTracker.lostDetectionsBuffer.AppendDets(renamedOld)

	matchMtx := fakeTracker.BuildMatchingMatrix(renamedOld, filteredNew)
	HA, err := hg.NewHungarianAlgorithm(matchMtx)
	test.That(t, err, test.ShouldBeNil)
	matches := HA.Execute()
	lostDetections := []objdet.Detection{}
	for idx, _ := range matches {
		if matches[idx] == -1 {
			lostDetections = append(lostDetections, renamedOld[idx])
		}
	}
	test.That(t, len(lostDetections), test.ShouldEqual, 1)
	checkLabel(t, lostDetections[0], LabelDet1) //we should be losing "fish"
	fakeTracker.lostDetectionsBuffer.AppendDets(lostDetections)

	// Rename from temporal matches. New det copies old det's label
	renamedNew, _ := fakeTracker.RenameFromMatches(matches, matchMtx, renamedOld, filteredNew)

	//Store stuffs
	fakeTracker.lastDetections = renamedNew
	fakeTracker.currDetections.mutex.Lock()
	fakeTracker.currDetections.detections = renamedNew
	fakeTracker.currDetections.mutex.Unlock()

	//Check output
	fakeTracker.currDetections.mutex.RLock()
	currDetections := fakeTracker.currDetections.detections
	fakeTracker.currDetections.mutex.RUnlock()
	test.That(t, len(currDetections), test.ShouldEqual, 1)
	checkLabel(t, currDetections[0], LabelDet0)

	//End of initialisation get new detections

	filteredNew = fd.fakeDetections() //get fish but somewhere else

	// Store oldDetection and lost detections in allDetections
	allDetections := fakeTracker.lastDetections
	for _, dets := range fakeTracker.lostDetectionsBuffer.detections {
		allDetections = append(allDetections, dets...)
	}

	// Build and solve cost matrix via Munkres' method
	matchMtx = fakeTracker.BuildMatchingMatrix(allDetections, filteredNew)
	HA, _ = hg.NewHungarianAlgorithm(matchMtx)
	matches = HA.Execute()
	lostDetections = []objdet.Detection{}
	for idx, _ := range fakeTracker.lastDetections {
		if matches[idx] == -1 {
			lostDetections = append(lostDetections, fakeTracker.lastDetections[idx])
		}
	}
	test.That(t, len(lostDetections), test.ShouldEqual, 1)
	checkLabel(t, lostDetections[0], LabelDet0) //now we loose the cat

	//}
	// Rename from temporal matches. New det copies old det's label
	renamedNew, _ = fakeTracker.RenameFromMatches(matches, matchMtx, allDetections, filteredNew)
	fakeTracker.lostDetectionsBuffer.AppendDets(lostDetections)

	// Store results
	fakeTracker.lastDetections = renamedNew
	fakeTracker.currDetections.mutex.Lock()
	fakeTracker.currDetections.detections = renamedNew
	fakeTracker.currDetections.mutex.Unlock()

	// Check output
	fakeTracker.currDetections.mutex.RLock()
	currDetections = fakeTracker.currDetections.detections
	fakeTracker.currDetections.mutex.RUnlock()
	test.That(t, len(currDetections), test.ShouldEqual, 1)
	test.That(t, currDetections[0].Label()[:len(LabelDet1)], test.ShouldEqual, LabelDet1)

	//Detecting a new cat, now we want to make sure that we don't have 2 "fish_zero"
	//when fish get lost
	filteredNew = fd.fakeDetections() //get cat again
	test.That(t, len(filteredNew), test.ShouldEqual, 1)
	checkLabel(t, filteredNew[0], LabelDet0)

	// Store oldDetection and lost detections in allDetections
	allDetections = fakeTracker.lastDetections
	for _, dets := range fakeTracker.lostDetectionsBuffer.detections {
		allDetections = append(allDetections, dets...)
	}

	// Build and solve cost matrix via Munkres' method
	matchMtx = fakeTracker.BuildMatchingMatrix(allDetections, filteredNew)
	HA, _ = hg.NewHungarianAlgorithm(matchMtx)
	matches = HA.Execute()
	lostDetections = []objdet.Detection{}
	for idx, _ := range fakeTracker.lastDetections {
		if matches[idx] == -1 {
			lostDetections = append(lostDetections, fakeTracker.lastDetections[idx])
		}
	}
	test.That(t, len(lostDetections), test.ShouldEqual, 1)
	checkLabel(t, lostDetections[0], LabelDet1) //loosing the fish

	//}
	// Rename from temporal matches. New det copies old det's label
	renamedNew, _ = fakeTracker.RenameFromMatches(matches, matchMtx, allDetections, filteredNew)
	test.That(t, len(fakeTracker.lostDetectionsBuffer.detections[0]), test.ShouldEqual, 1)
	checkLabel(t, fakeTracker.lostDetectionsBuffer.detections[0][0], LabelDet1) //check if there used to be fish
	test.That(
		t,
		fakeTracker.lostDetectionsBuffer.detections[0][0].BoundingBox().Min,
		test.ShouldResemble,
		image.Pt(20, 20),
	)
	test.That(
		t,
		fakeTracker.lostDetectionsBuffer.detections[0][0].BoundingBox().Max,
		test.ShouldResemble,
		image.Pt(30, 30),
	)
	fakeTracker.lostDetectionsBuffer.AppendDets(lostDetections)
	//check if the last fish_0 has been deleted
	test.That(t, len(fakeTracker.lostDetectionsBuffer.detections[0]), test.ShouldEqual, 0)

	//check if the new fish is actually new (updated bbox)
	test.That(t, len(fakeTracker.lostDetectionsBuffer.detections[2]), test.ShouldEqual, 1)
	checkLabel(t, fakeTracker.lostDetectionsBuffer.detections[2][0], LabelDet1)
	test.That(
		t,
		fakeTracker.lostDetectionsBuffer.detections[2][0].BoundingBox().Min,
		test.ShouldResemble,
		image.Pt(22, 22),
	)
	test.That(
		t,
		fakeTracker.lostDetectionsBuffer.detections[2][0].BoundingBox().Max,
		test.ShouldResemble,
		image.Pt(33, 33),
	)

	test.That(
		t,
		fakeTracker.lostDetectionsBuffer.detections[2][0].BoundingBox().Max,
		test.ShouldNotResemble,
		image.Pt(20, 20),
	)

	test.That(
		t,
		fakeTracker.lostDetectionsBuffer.detections[2][0].BoundingBox().Max,
		test.ShouldNotResemble,
		image.Pt(30, 30),
	)

	// Store results
	fakeTracker.lastDetections = renamedNew
	fakeTracker.currDetections.mutex.Lock()
	fakeTracker.currDetections.detections = renamedNew
	fakeTracker.currDetections.mutex.Unlock()

	// Check output
	fakeTracker.currDetections.mutex.RLock()
	currDetections = fakeTracker.currDetections.detections
	fakeTracker.currDetections.mutex.RUnlock()
	test.That(t, len(currDetections), test.ShouldEqual, 1)
	checkLabel(t, currDetections[0], LabelDet0)
}
