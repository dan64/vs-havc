# HAVC — Model Comparison: Detailed Test Results

This document contains the full per-image **CIEDE2000** results of the colorization models supported by HAVC. For an introduction to the compared models and a high-level summary of the findings, see the [main README](../README.md#comparison-of-models).

## Comparison Methodology

To compare the models I decided to use a metric being able to consider the _perceptual non-uniformities_ in the evaluation of color difference between images. These non-uniformities are important because the human eye is more sensitive to certain colors than others. Over time, The International Commission on Illumination (**CIE**) has proposed increasingly advanced measurement models to measure the color distance taking into account the _human color perception_, that they called **dE**. One of the most advanced is the [CIEDE2000](https://en.wikipedia.org/wiki/Color_difference#CIEDE2000) method, that I decided to use as _color similarity metric_ to compare the models. The final results are shown in the tables below (test image can be seen by clicking on the test number).

The calculation of **dE** with the **CIEDE2000** method was obtained by leveraging on the computational code available in [ColorMine](https://github.com/MasterPieceCode/Mozaic/tree/master/ColorMine).

## Test Set #1

The compared models in this test set are:

- **D+D**: DeOldify (with model _Video_ & render_factor = 24) + DDColor (with model _Artistic_ and render_factor = 24)
- **DD**: DDColor (with model _Artistic_ and render_factor = 24, equivalent to input_size = 384)
- **DS**: DeOldify (with model _Stable_ & render_factor = 24)
- **DV**: DeOldify (with model _Video_ & render_factor = 24)
- **T241**: ColTran + TensorFlow 2.4.1 model as shown in [Habr](https://habr.com/en/companies/ruvds/articles/568426/)

| Test # | D+D | DD | DS | DV  | T241 |
|---|---|---|---|---|---|
|[01](https://github.com/dan64/vs-havc/blob/main/test_images/Image_01_test.jpg) | 10.7 | **8.7** | 8.8 | 12.7 | 15.7 |
|[02](https://github.com/dan64/vs-havc/blob/main/test_images/Image_02_test.jpg) | 11.8 | **11.7** | 12.7 | 12.7 | 15.9 |
|[03](https://github.com/dan64/vs-havc/blob/main/test_images/Image_03_test.jpg) | 5.5 | **3.8** | 5.6 | 7.6 | 9.9 |
|[04](https://github.com/dan64/vs-havc/blob/main/test_images/Image_04_test.jpg) | 6.2 | 8.5 | **4.6** | 5.3 | 9.0 |
|[05](https://github.com/dan64/vs-havc/blob/main/test_images/Image_05_test.jpg) | **6.6** | 8.4 | 8.8 | 8.6 | 12.5 |
|[06](https://github.com/dan64/vs-havc/blob/main/test_images/Image_06_test.jpg) | 10.2 | **9.9** | 10.6 | 11.2 | 16.4 |
|[07](https://github.com/dan64/vs-havc/blob/main/test_images/Image_07_test.jpg) |**6.5** | 6.7 | 6.8 | 7.7 | 10.2 |
|[08](https://github.com/dan64/vs-havc/blob/main/test_images/Image_08_test.jpg) | 6.7 | **6.4** | 7.5 | 8.3 | 9.9 |
|[09](https://github.com/dan64/vs-havc/blob/main/test_images/Image_09_test.jpg) | 11.7 | **11.7** | 15.2 | 13.8 | 16.5 |
|[10](https://github.com/dan64/vs-havc/blob/main/test_images/Image_10_test.jpg) | **7.8** | 8.0 | 9.1 | 8.4 | 9.5 |
|[11](https://github.com/dan64/vs-havc/blob/main/test_images/Image_11_test.jpg) | **7.5** | 8.0 | 8.0 | 7.8 | 14.8 |
|[12](https://github.com/dan64/vs-havc/blob/main/test_images/Image_12_test.jpg) | 7.7 | **7.6** | 8.6 | 7.8 | 13.7 |
|[13](https://github.com/dan64/vs-havc/blob/main/test_images/Image_13_test.jpg) | **11.8** | 11.9 | 14.2 | 13.7 | 16.8 |
|[14](https://github.com/dan64/vs-havc/blob/main/test_images/Image_14_test.jpg) | 5.3 | 5.2 | **4.4** | 5.3 | 7.2 |
|[15](https://github.com/dan64/vs-havc/blob/main/test_images/Image_15_test.jpg) | 8.2 | **7.3** | 10.7 | 10.6 | 15.7 |
|[16](https://github.com/dan64/vs-havc/blob/main/test_images/Image_16_test.jpg) | 12.0 | 12.3 | **9.8** | 12.7 | 19.7 |
|[17](https://github.com/dan64/vs-havc/blob/main/test_images/Image_17_test.jpg) | 11.1 | **10.2** | 11.6 | 12.4 | 16.7 |
|[18](https://github.com/dan64/vs-havc/blob/main/test_images/Image_18_test.jpg) | **6.7** | 9.3 | 7.2 | 8.6 | 13.1 |
|[19](https://github.com/dan64/vs-havc/blob/main/test_images/Image_19_test.jpg) | **3.7** | 4.4 | 4.7 | 3.9 | 4.6 |
|[20](https://github.com/dan64/vs-havc/blob/main/test_images/Image_20_test.jpg) | 8.7 | 10.1 | **6.9** | 9.2 | 11.0 |
|[21](https://github.com/dan64/vs-havc/blob/main/test_images/Image_21_test.jpg) | **6.9** | 6.9 | 8.1 | 8.4 | 10.4 |
|[22](https://github.com/dan64/vs-havc/blob/main/test_images/Image_22_test.jpg) | **11.5** | 11.8 | 13.3 | 12.2 | 12.7 |
|[23](https://github.com/dan64/vs-havc/blob/main/test_images/Image_23_test.jpg) | **5.6** | 7.1 | 11.4 | 8.8 | 11. |
|**Avg(dE)** | **8.3** | **8.5** | **9.1** | **9.5** | **12.7** |

As it is possible to see the model that performed better is the **D+D** model (which I called _HAVC ddelodify_ because is using both _DeOldify_ and _DDColor_). This model was the best model in 10 tests out of 23. Also the **DD** model performed well but there were situations where the **DD** model provided quite bad colorized images like in [Test #23](https://github.com/dan64/vs-havc/blob/main/test_images/Image_23_test.jpg) and the combination with the DeOldify allowed to significantly improve the final image. In effect the average distance of **DD** was **8.5** while for **DV** was **9.5**, given that the 2 models were weighted at 50%, if the images were positively correlated a value **9.0** would have been expected, instead the average distance measured for **D+D** was **8.3**, this implies that the 2 models were able to compensate each other.
Conversely, the **T241** was the model that performed worse with the greatest average difference in colors. Finally, the quality of DeOldify models was similar, being **DS** slightly better than **DV** (as expected).

## Tests Set #2

Given the goodness of **CIEDE2000** method to provide a reliable estimate of _human color perception_, I decided to provide an additional set of tests including some of the cases not considered previously.

The models added are:

- **DA**: DeOldify (with model _Artistic_ & render_factor = 30)
- **DDs**: DDColor (with model _ModelScope_ and input_size = 384)
- **DS+DD**: DeOldify (with model _Stable_ & render_factor = 30) + DDColor (with model _Artistic_ and render_factor = 24)
- **DA+DDs**: DeOldify (with model _Artistic_ & render_factor = 30) + DDColor (with model _ModelScope_ and render_factor = 24)
- **DA+DD**: DeOldify (with model _Artistic_ & render_factor = 30) + DDColor (with model _Artistic_ and render_factor = 24)

The results of this additional tests set are shown in the table below (test image can be seen by clicking on the test number)

| Test # | DS+DD | DA+DDs | DA+DD | DDs  | DA |
|---|---|---|---|---|---|
|[01](https://github.com/dan64/vs-havc/blob/main/test_images_ex/Image_01_test_ex.jpg) | 7.7 | **7.5** | 8.2 | 8.2 | 8.6 |
|[02](https://github.com/dan64/vs-havc/blob/main/test_images_ex/Image_02_test_ex.jpg) | 11.8 | **11.4** | 11.9 | 11.6 | 13.2 |
|[03](https://github.com/dan64/vs-havc/blob/main/test_images_ex/Image_03_test_ex.jpg) | 4.5 | 4.2 | **3.9** | 4.5 | 4.2 |
|[04](https://github.com/dan64/vs-havc/blob/main/test_images_ex/Image_04_test_ex.jpg) | 5.9 | **5.1** | 6.0 | 6.6 | 5.9 |
|[05](https://github.com/dan64/vs-havc/blob/main/test_images_ex/Image_05_test_ex.jpg) | **6.4** | 6.5 | 6.7 | 9.5 | 9.0 |
|[06](https://github.com/dan64/vs-havc/blob/main/test_images_ex/Image_06_test_ex.jpg) | 10.0 | 10.0 | 10.3 | **9.5** | 11.4 |
|[07](https://github.com/dan64/vs-havc/blob/main/test_images_ex/Image_07_test_ex.jpg) | **6.1** | 7.3 | 6.6 | 8.1 | 8.0 |
|[08](https://github.com/dan64/vs-havc/blob/main/test_images_ex/Image_08_test_ex.jpg) | **6.2** | 8.1 | 7.3 | 8.1 | 9.4 |
|[09](https://github.com/dan64/vs-havc/blob/main/test_images_ex/Image_09_test_ex.jpg) | 12.7 | **11.3** | 11.5 | 12.5 | 13.3 |
|[10](https://github.com/dan64/vs-havc/blob/main/test_images_ex/Image_10_test_ex.jpg) | 8.1 | 7.7 | 8.0 | **7.1** | 9.0 |
|[11](https://github.com/dan64/vs-havc/blob/main/test_images_ex/Image_11_test_ex.jpg) | **7.2** | 7.3 | 7.4 | 8.6 | 7.9 |
|[12](https://github.com/dan64/vs-havc/blob/main/test_images_ex/Image_12_test_ex.jpg) | 8.0 | 7.1 | 8.0 | **6.5** | 9.3 |
|[13](https://github.com/dan64/vs-havc/blob/main/test_images_ex/Image_13_test_ex.jpg) | 12.0 | **11.7** | 12.0 | 11.8 | 13.8 |
|[14](https://github.com/dan64/vs-havc/blob/main/test_images_ex/Image_14_test_ex.jpg) | **4.5** | 4.6 | 4.8 | 5.8 | 4.8 |
|[15](https://github.com/dan64/vs-havc/blob/main/test_images_ex/Image_15_test_ex.jpg) | 8.3 | **8.1** | 8.9 | 8.2 | 12.2 |
|[16](https://github.com/dan64/vs-havc/blob/main/test_images_ex/Image_16_test_ex.jpg) | 10.6 | 10.5 | 10.7 | 12.5 | **9.9** |
|[17](https://github.com/dan64/vs-havc/blob/main/test_images_ex/Image_17_test_ex.jpg) | **10.8** | 12.1 | 11.4 | 12.3 | 13.5 |
|[18](https://github.com/dan64/vs-havc/blob/main/test_images_ex/Image_18_test_ex.jpg) | 6.7 | 7.1 | **6.1** | 11.1 | 7.2 |
|[19](https://github.com/dan64/vs-havc/blob/main/test_images_ex/Image_19_test_ex.jpg) | **3.5** | 4.6 | 4.5 | 5.1 | 7.1 |
|[20](https://github.com/dan64/vs-havc/blob/main/test_images_ex/Image_20_test_ex.jpg) | 8.0 | 8.1 | 8.2 | 9.3 | **7.6** |
|[21](https://github.com/dan64/vs-havc/blob/main/test_images_ex/Image_21_test_ex.jpg) | 6.9 | **6.7** | 7.1 | 7.1 | 9.0 |
|[22](https://github.com/dan64/vs-havc/blob/main/test_images_ex/Image_22_test_ex.jpg) | 12.1 | 11.0 | **10.9** | 12.1 | 11.2 |
|[23](https://github.com/dan64/vs-havc/blob/main/test_images_ex/Image_23_test_ex.jpg) | 6.2 | 6.3 | **6.0** | 7.8 | 10.2 |
|**Avg(dE)** | **8.0** | **8.0** | **8.1** | **8.9** | **9.4** |

First of all, it should be noted that the individual models added (**DA** for _DeOldify_ and **DDs** for _DDColor_) performed worse than the individual models tested in the previous analysis (**DS** for _DeOldify_ and **DD** for _DDColor_). Conversely all combinations of _DeOldify_ and _DDColor_ performed well. Confirming the positive impact on the final result, already observed in the previous analysis, obtained by combining the 2 models.
