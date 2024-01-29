// Package object_tracker implements an object tracker as a Viam vision service
package object_tracker

import (
	objdet "go.viam.com/rdk/vision/objectdetection"
	"image"
)

// IOU returns the intersection over union of 2 rectangles
func IOU(r1, r2 *image.Rectangle) float64 {
	intersection := r1.Intersect(*r2)
	if intersection.Empty() {
		return 0
	}
	union := r1.Union(*r2)
	return float64(intersection.Dx()*intersection.Dy()) / float64(union.Dx()*union.Dy())
}

// PredictNextFrame assumes we have two rectangles on frames n-1 and n. We use those
// to predict the rectangle on frame n+1
func PredictNextFrame(old, curr image.Rectangle) image.Rectangle {

	// Calculate the Vx and Vy based on a linear velocity vibe
	oldCX, oldCY := float64((old.Min.X+old.Max.X)/2), float64((old.Min.Y+old.Max.Y)/2)
	currCX, currCY := float64((curr.Min.X+curr.Max.X)/2), float64((curr.Min.Y+curr.Max.Y)/2)
	newCx, newCy := currCX+(currCX-oldCX), currCY+(currCY-oldCY) // add single frame velocity

	x0, x1 := newCx-float64(curr.Dx()/2), newCx+float64(curr.Dx()/2)
	y0, y1 := newCy-float64(curr.Dy()/2), newCy+float64(curr.Dy()/2)

	return image.Rect(int(x0), int(y0), int(x1), int(y1))

}

// BuildMatchingMatrix sets up a cost matrix for the Hungarian algorithm
// In this implementation, cost is -IOU between bboxes (b/c solver will find min)
// May need to edit to do the actual predictions before comparing to newDets
func BuildMatchingMatrix(oldDetections, newDetections []objdet.Detection) [][]float64 {
	h, w := len(oldDetections), len(newDetections)
	matchMtx := make([][]float64, h)
	for i, oldD := range oldDetections {
		row := make([]float64, w)
		for j, newD := range newDetections {
			row[j] = -IOU(oldD.BoundingBox(), newD.BoundingBox())
		}
		matchMtx[i] = row
	}
	return matchMtx
}

// https://github.com/charles-haynes/munkres/  <-- THIS ONE!
