// Package object_tracker does the object tracking or whateva
package object_tracker

import (
	"fmt"
	"image"
	"math"
)

func IOU(r1, r2 image.Rectangle) float64 {
	intersection := r1.Intersect(r2)
	if intersection.Empty() {
		return 0
	}
	union := r1.Union(r2)
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

// IOU2 returns the IOU assuming bounding boxes are [x1, y1, x2, y2].
func PredictNextFrame(old, curr image.Rectangle) []float64 {

	// Calculate the Vx and Vy based on a linear velocity vibe
	oldCX, oldCY := float64((old.Min.X+old.Max.X)/2), float64((old.Min.Y+old.Max.Y)/2)
	currCX, currCY := float64((curr.Min.X+curr.Max.X)/2), float64((curr.Min.Y+curr.Max.Y)/2)
	vx, vy := currCX-oldCX, currCY-oldCY // single frame velocity

	fmt.Println(vx, vy)

	return []float64{3.1415926535897932384626433832795028841971693993751058209}
}

// https://github.com/arthurkushman/go-hungarian
