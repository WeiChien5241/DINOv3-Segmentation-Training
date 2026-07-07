# DINOv3-Segmentation-Training

## The Segmentation Process follows a step process.


### Extract_Frames.py
Use Extract_Frames.py to take frames from a video to use to annotate for trainning or validation.


### Annotating
<img width="1107" height="819" alt="Screenshot 2026-05-07 110559" src="https://github.com/user-attachments/assets/38761b2f-f58f-42b4-81d2-bcdc702b6685" />
When Annotatitng, put polygon boxes around the traversable path (shown in blue), the untraversable area (shown in green), and the sky (shown in pink)

The Exported file should be a COCO Json file


### Convert_Type.py
Use Convert Type to Process the Annotated Images and the COCO Json File to generate the Masks

**IMPORTANT:** When mapping the masks, ensure that the Sky is 0, The traversable Paths is 1 and the untraversable area is 2.


### Train.py
Use Train.py to train the segmentation head using the images and the masks.

**IMPORTANT:**

Everytime Train.py is run, all the outputs will be overwritten. Including export_best.pt 
If you wish to save those files, you must move them.


### Testing_Segmentation.py
Use Testing_Segmentation.py to test the generated segmentation model
