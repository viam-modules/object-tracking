// Package main is a module which serves the mybase custom model.
package main

import (
	"go.viam.com/rdk/resource"
	"go.viam.com/rdk/services/vision"

	"go.viam.com/rdk/module"

	"github.com/viam-modules/object-tracking/object_tracker"
)

func main() {
	module.ModularMain(resource.APIModel{API: vision.API, Model: object_tracker.Model})
}
