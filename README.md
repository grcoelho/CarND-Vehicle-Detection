Project: Vehicle Detection - First Review
===================


The goals of this project are:
* Use the front face image from a video to detect Vehicles around.
* Draw Boxes around the vehicles
* Merge this project with Advanced Lanes Finding

To accomplish that, I checked this video from Ryan Keenan with a walkthrough for the project:
[Youtube Link](https://www.youtube.com/watch?v=P2zwrTM8ueA&feature=youtu.be&t=2821)



Step-by-Step Approach
-------------


Most of the code came from the Vehicle Detection and Tracking lesson. Most are examples and quizzes, so I will stick to the modifications and tunning I made to accomplish the objective.

Here is a list of functions used from the lesson as is:
- bin_spatial()
- get_hog_features()
- slide_window()
- color_hist()
- extract_features()
-  draw_boxes()
- single_img_features()
- search_windows()
- add_heat()
- apply_threshold()
- draw_labeled_bboxes()


#### <i class="icon-file"></i> Car and Non-Car Classifier
Like we did in the class, I just used a SVC Classifier. I decided to use the full database provided to train this classifier (8792 Vehicles and  8968 non Vehicles).
I took this approach to have the maximum data for training available, and I tested with smaller datasets and found that they were not that good. I merged the files on the folder changing the file names, so its not necessary to randomize the training data on the algorithm.
I tested the classifier on images first and then in the test video. The first parameters tunning was just something to get it going on videos, because I found out that its much easier to make the classifier to work as you want on images than in videos (changes in brightness, shadows and other noise make it a bit unpredictable).

#### <i class="icon-file"></i> Parameters Tunning

About the parameters, when using videos I choosen by try and error. I adjusted each one and find out wich detects more squares around the car. The color channel I decided to use was YCrCb. 
I kept HOG Orients in 9, like we did on the class (I didnt find much difference while changing this parameter, so I went to basics). Like Orients, pixels per cell were 8 and 2 cells per block. Hog channel was kept in zero, like in the class.
Spatial size was (16, 16), and Hist Bin was 16 too.
To remove the horizon, I applied the following limitations. I also applied one to x_start_stop as a tip from my mentor. She told me that this could be useful to remove false positives around the bushes near the road.

    y_start_stop = [350, 700] # Min and max in y to search in slide_window()
    x_start_stop = [200, 1200] # Min and max in x to search in slide_window()

With that, I decided to adjust the window sizes to map more effectivelly the image.

    windows = slide_window(image, x_start_stop=x_start_stop, y_start_stop=y_start_stop, 
                    window_sizes=(64, 96, 128, 160), xy_overlap=(0.75, 0.75))
Now I pass the window sizes as just one integer number for each one I want to cover. In the configuration above, we have a first pass with 64x64 windows, then another one with 96x96, another one with 128x128 and the last one with 160x160.  The modification on the original function was the new for loop to cover all window sizes passed. I choosen 50% of xy overlap, because it work with this value and increasing that would increase the processing need to process the bigger number of windows generated.

    def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    window_sizes=(64, 96,128), xy_overlap=(0.5, 0.5)):
    # Initialize a list to append window positions to
    window_list = []
    for size in window_sizes:
        
        xy_window = [size,size]
        # If x and/or y start/stop positions not defined, set to image size
        if x_start_stop[0] == None:
            x_start_stop[0] = 0
        if x_start_stop[1] == None:
            x_start_stop[1] = img.shape[1]
        if y_start_stop[0] == None:
            y_start_stop[0] = 0
        if y_start_stop[1] == None:
            y_start_stop[1] = img.shape[0]
        # Compute the span of the region to be searched    
        xspan = x_start_stop[1] - x_start_stop[0]
        yspan = y_start_stop[1] - y_start_stop[0]
        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
        # Compute the number of windows in x/y
        nx_windows = np.int(xspan/nx_pix_per_step) - 1
        ny_windows = np.int(yspan/ny_pix_per_step) - 1
        # Loop through finding x and y window positions
        # Note: you could vectorize this step, but in practice
        # you'll be considering windows one by one with your
        # classifier, so looping makes sense
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                startx = xs*nx_pix_per_step + x_start_stop[0]
                endx = startx + xy_window[0]
                starty = ys*ny_pix_per_step + y_start_stop[0]
                endy = starty + xy_window[1]

                # Append window position to list
                window_list.append(((startx, starty), (endx, endy)))
        # Return the list of windows        
        
    return window_list

By trial and error, and with some tips from Slack about what worked for others, I found out that more passes through the image with different image sizes would generate more data (even generating more garbage too). If I have a lot of data with garbage (that appears not always in the same place), and the big number of squares over cars, It would be easier to filter boxes that appear more in a part of the window and eliminate the rest. 
Here is an image of my result running with this configuration without filtering.

![enter image description here](https://lh3.googleusercontent.com/izJlpO_7GcTrjrDv4qDGf7pEL79fLi-LexSAUMz-0tlT1alqgVZoMTQ2Jr843osDr38_ePuDXg=s0 "lot of boxes.png")

As we can see, its an easy image (not that much garbage) and I have a lot of boxes over the cars. This will result in a heat map with a lot of "heat" over them.

#### <i class="icon-file"></i> Filtering

I used the same idea from the class here. Draw a heap map and threshold the number of overlapping boxes. But here I have a lot more boxes, so my threshold are a lot bigger. On the class we only kept results with more than one overlapping boxes, on the following image I kept only with more than 3 overlapping boxes. The result is a clean heatmap showing clearly the position of those two cars.
![enter image description here](https://lh3.googleusercontent.com/YEPhJXS_m2CgpGgmterTIREIc1Hz_VLm2b11mj6WHkTqfw0ZhUtFPLlKMFPqrfqdtgzSEZHNpg=s0 "heatmap.png")

During videos, things are a bit more complex. A lot of false detections appear from frame to frame, and a lot of garbage is generated. To deal with that, I increased the threshold limit to 5 (keeping only if more than five boxes overlaps the region), and start averaging between frames. So even if a false positive appears and vanishes from frame to frame, it would have less impact on the result. 
The algorithm to do that was pretty simple. Just kept 20 frames of history and averaged between them. 

    # Apply threshold to help remove false positives
        heat = apply_threshold(heat,3)
        
    if frame_vdt == 1:
            heat_1 = heat
            heat_2 = np.zeros_like(image[:,:,0]).astype(np.float)
            heat_3 = np.zeros_like(image[:,:,0]).astype(np.float)
            heat_4 = np.zeros_like(image[:,:,0]).astype(np.float)
            heat_5 = np.zeros_like(image[:,:,0]).astype(np.float)
            heat_6 = np.zeros_like(image[:,:,0]).astype(np.float)
            heat_7 = np.zeros_like(image[:,:,0]).astype(np.float)
            heat_8 = np.zeros_like(image[:,:,0]).astype(np.float)
            heat_9 = np.zeros_like(image[:,:,0]).astype(np.float)
            heat_10 = np.zeros_like(image[:,:,0]).astype(np.float)
            heat_11 = np.zeros_like(image[:,:,0]).astype(np.float)
            heat_12 = np.zeros_like(image[:,:,0]).astype(np.float)
            heat_13 = np.zeros_like(image[:,:,0]).astype(np.float)
            heat_14 = np.zeros_like(image[:,:,0]).astype(np.float)
            heat_15 = np.zeros_like(image[:,:,0]).astype(np.float)
            heat_16 = np.zeros_like(image[:,:,0]).astype(np.float)
            heat_17 = np.zeros_like(image[:,:,0]).astype(np.float)
            heat_18 = np.zeros_like(image[:,:,0]).astype(np.float)
            heat_19 = np.zeros_like(image[:,:,0]).astype(np.float)
            heat_20 = np.zeros_like(image[:,:,0]).astype(np.float)

        else:
            heat_20 = heat_19
            heat_19 = heat_18
            heat_18 = heat_17
            heat_17 = heat_16
            heat_16 = heat_15
            heat_15 = heat_14
            heat_14 = heat_13
            heat_13 = heat_12
            heat_12 = heat_11
            heat_11 = heat_10
            heat_10 = heat_9
            heat_9 = heat_8
            heat_8 = heat_7
            heat_7 = heat_6
            heat_6 = heat_5
            heat_5 = heat_4
            heat_4 = heat_3
            heat_3 = heat_2
            heat_2 = heat_1
            heat_1 = heat

        k1 = heat_20+ heat_19+heat_18+heat_17+heat_16+heat_15+heat_14+ heat_13+heat_12+heat_11+heat_10+heat_9+heat_8+heat_7+heat_6+heat_5+heat_4+heat_3+heat_2+heat_1
 
        k2 = k1 / 20
        
        heat = k2 #temp
        
        # Apply threshold to help remove false positives
        heat = apply_threshold(heat,5)

I understand that this is a pretty idiot code. But it worked fairly well on all the video, eliminating false positives from frame to frame and smoothing the result. I tried using weights on frames, but doing just the average worked better.
I also included another thresholding after taking the average from frame to frame, so I could avoid storing garbage from frame to frame. The threshold was just 3, since this was less related to the final result than the final average.
Here is an example of frame to frame averaging (I just plotted 3 frames, and the result of the averaging).

![enter image description here](https://lh3.googleusercontent.com/8zX7pC7u46YKgRiEDhb4sUEb2GE-r_I0n2earFznFN09TmaFsIfH-GILe6bwQFjT_hPBYVbSKg=s0 "frame-to-frame-average.png")

#### <i class="icon-file"></i> Post Processing

With the averaged heat map in hands, I just used the label function to get the final boxes and draw_labeled_bboxes to draw the final boxes. As you can see here, to merge this project with the last one, I just added the lane_lines final image here and draw the lines on an already processed image (I put both algorithms running on the same Jupyter Notebook).

        # Visualize the heatmap when displaying    
        heatmap = np.clip(heat, 0, 255)
        
        from scipy.ndimage.measurements import label
        labels = label(heatmap)
       
        # Draw bounding boxes on a copy of the image
        draw_img = draw_labeled_bboxes(np.copy(lane_lines/255.0), labels)


Reflections
-------------
I think that this project was more interesting than the earlier one, because It does not include that much image bytes brushing to get an result. Using the SVC Classifier was great and gave an easy result on detection (I still think the results could be better with a Keras structure, because we would have more parameters to tune, like the architeture and deep of the network). I used a lot of data to train the classifier, and get 97% of accuracy on test set. I think that data augmentation could be a good idea not to just get more data, but to have data with different shadowing conditions, different positions and different light and color conditions.
About the processing time, my algorithm is really heavy. Speacially the windowing part. 
I have to take much samples through a lot of windows on the image to get a bunch of boxes so I can filter.
I tried with less windows, and found that usually the amount of false positives was close to the positive, restricting the possibility to filter. I think that a better classification would help on that, reducing the false positives and reducing the need of heavy filtering. This is something to try and get an improvement (as far as I measured, it took around 4 seconds to run windowing, this was the bottleneck).
I think the next step would be to try with another classifier and reduce the amount of windows.
Another issue that I found is that the algorithm is not using full CPU capacity. It stays with around 30%, and I think this may be related to single core processing. This is something to investigate, and can give a notable improvement in performance.
About the result, I found it to detect the cars properly. Some false positives still appear, but they are small and vanishes soon. I tried to implement another frame to frame function to ignore false positives, but my tried prove to get worse results than this one. Another try was to implement a function to ignore boxes with small area inside, but this proves to remove some positive cars in a far distance, so I gave up from this idea and decided that Its better to have some noise than to remove positive cars detections.
By the end, I think this algorithm would probably fail if the road conditions changes. I saw that a lot of training data for not cars include images from the video in question, so in a new video its probable to perform not as good as in the project video because of the training conditions on the dataset provided. Another possible failure would be if some dirt or even rain droplets stay in focus of the camera, since they can be detected by the classifier and end up appearing on the final heat map (since they will be constantly with the same shape and position).



