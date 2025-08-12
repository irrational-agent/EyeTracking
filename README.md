## EyeTracking
The fork aims to fix the memory usage issues during training, refactor ML code and improve convenience of training.
I noticed many problems trying to use the original project (my GPU is worse), therefore decided to fix the training problems of the original work.
However, I really like what the author did so I decided to make my improvements to this project.

---
This is a fork of the repository: https://github.com/ryan9411vr/EyeTracking.
View the original repository for the detailed usage tutorial. 

View [dev branch](https://github.com/irrational-agent/EyeTracking/tree/dev) to see recent progress.

--- 

Work in progress...

Issues found in the original:
- Tested on RTX 3070TI + 32GB RAM, does not work (memory issues).
- Tested on RTX 5070 + 24GB RAM, does not work (memory issues).
- Images are read as RGB and the model uses 3 channels, however the training data is grayscale.
- All the data is loaded into RAM at once. Therefore, original work has hardcoded limits. Loading data by batches should unlock the ability to train larger datasets.
- Loading data from the database is very slow. I would just remove the database.
- Dataset uses both eye openness and eye gaze data (merged) and you can't just clear one type of data to record it again. 

Todo list:
- [x] Refactor training code.
- [x] Make data load gradually, by batches. In this case, much, much less memory is used.
- [ ] Apply changes to all training files.
- [ ] Remove database usage. Switch to regular folders (I prefer this for faster training because loading the database each time is terrible).
- [ ] Separate eye gaze data and eye openness data. (to record data separately. Now if you mess up data in the dataset, both gaze and openness data will be bad).
- [ ] Make converter script use the same environment?
- [ ] Automate target movement in Unity app.
- [ ] Recording indicators in Unity app.
- [ ] Display recorded data samples count in Unity app.
- [ ] Better eye openness visualization in Unity app?
- [ ] Improve data augmentation.
- [ ] Add dropout layers?

Future plans:
- [ ] An ability to merge different datasets? Pretrained networks?
