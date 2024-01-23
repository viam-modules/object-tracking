// Package object_tracker does the object tracking or whateva
package object_tracker

import (
	objdet "go.viam.com/rdk/vision/objectdetection"
	"image"
	"math"
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

// IOU2 returns the IOU assuming bounding boxes are [x1, y1, x2, y2].
func IOU2(r1, r2 [4]float64) float64 {
	// Find intersection
	intx1, intx2 := math.Max(r1[0], r2[0]), math.Min(r1[2], r2[2])
	inty1, inty2 := math.Max(r1[1], r2[1]), math.Min(r1[3], r2[3])

	// Calculate areas
	areaInt := (intx2 - intx1) * (inty2 - inty1)
	area1 := (r1[2] - r1[0]) * (r1[3] - r1[1])
	area2 := (r2[2] - r2[0]) * (r2[3] - r2[1])

	// Return intersection over union
	return areaInt / (area1 + area2 - areaInt)
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

// https://github.com/arthurkushman/go-hungarian
// https://github.com/oddg/hungarian-algorithm/
// https://github.com/charles-haynes/munkres/  <-- THIS ONE!
