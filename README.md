# object-tracker

Viam provides an `object-tracker` model of the [vision service](/services/vision) to connect objects that are the same across time.

Configure this vision service as a [modular resource](https://docs.viam.com/modular-resources/) on your robot to transform your camera into an object tracking camera!

## Getting started

The first step is to configure a camera on your robot.  [Here](https://docs.viam.com/components/camera/webcam/) is an example of how to configure a webcam. The next step is to configure a vision service to use as a detector.  Remember the names given to the camera and detector, it will be important later.

> [!NOTE]
> Before configuring your camera or vision service, you must [create a robot](https://docs.viam.com/manage/fleet/robots/#add-a-new-robot).

## Configuration

Navigate to the **Config** tab of your robot’s page in [the Viam app](https://app.viam.com/). Click on the **Services** subtab and click **Create service**. Select the `vision` type, then select the `object-tracker` model. Enter a name for your service and click **Create**.

### Example Configuration

```json
{
  "modules": [
    {
      "type": "registry",
      "name": "viam_object-tracker",
      "module_id": "viam:object-tracker",
      "version": "latest"
    }
  ],
  "services": [
    {
      "name": "myObjectTracker",
      "type": "vision",
      "namespace": "rdk",
      "model": "viam:vision:object-tracker",
      "attributes": {
        "detector_name": "myDetector",
        "camera_name": "myCam",
        "max_frequency_hz": 20,
        "chosen_labels": {
          "scissors": 0.2,
          "dog": 0.3,
          "person": 0.7
        }
      }
    }
  ]
}

```

> [!NOTE]
> For more information, see [Configure a Robot](https://docs.viam.com/manage/configuration/).

### Attributes

The following attributes are available for `viam:vision:object-tracker` vision services:

| Name                  | Type               | Inclusion | Description                                                                                                                                                                                |
|-----------------------|--------------------| --------- |--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `camera_name`         | string             | **Required** | The name of the camera configured on your robot.                                                                                                                                           |
| `detector_name`       | string             | **Required** | The name of the detector (vision service) configured on your robot.                                                                                                                        |
| `min_confidence`      | float64            | **Optional** | A number between 0-1. Any detection with a confidence below this number will not be tracked. Default = 0.2                                                                                 |
| `max_frequency_hz`    | float64            | **Optional** | The fastest frequency (in Hz) that the model should run in. Default = 10.                                                                                                                  |
| `chosen_labels`       | map[string]float64 | **Optional** | A list of class names (string) and confidence scores (float[0-1]) such that **only** detections with a class name in the list and a confidence above the corresponding score are included. |
| `trigger_cool_down_s` | float64            | **Optional** | The duration (in seconds) before the trigger goes back to `empty`. Default = 5.                                                                                                            |
| `buffer_size`         | int                | **Optional** | Size of the buffer that stores lost bounding boxes. Default = 30. Min = 1. Max = 256.                                                                                                      |

### Example Attributes

```json
{
  "detector_name": "myDetector",
  "camera_name": "myCam",
  "max_frequency_hz": 20,
  "chosen_labels": {
    "scissors": 0.2,
    "dog": 0.3,
    "person": 0.7
  }
}
```

### Usage

This module is made for use with the following methods of the [vision service API](https://docs.viam.com/services/vision/#api):
- `GetProperties()`
- [`GetDetections()`](https://docs.viam.com/services/vision/#getdetections)
- [`GetDetectionsFromCamera()`](https://docs.viam.com/services/vision/#getdetectionsfromcamera)
- [`GetClassificationsFromCamera()`](https://docs.viam.com/services/vision/#getclassificationsfromcamera)
- `CaptureAll()`
- `DoCommand()`


The module will return a list of detections. The bounding box and `confidence` of each detection will be as detected by the underlying detector that was passed to the object-tracking module.  The new `class_name` will be: "< old `class_name`>_N_YYYYMMDD_HHMMSS", where the object is the Nth of it's class and was originally seen at the time/date indicated by YYYYMMDD_HHMMSS.


## Visualize

Once the `viam:vision:object-tracker` modular service is in use, configure a [transform camera](https://docs.viam.com/components/camera/transform/) detections appear in your robot's field of vision.
