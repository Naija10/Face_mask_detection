from tensorflow import keras
import numpy as np
import cv2

dataset=np.load("DATASETS/TRAIN.npy",allow_pickle=True)
classes=["mask","nomask"]
test_inputs=[]
test_targets=[]

for image ,target in dataset[3500:]:
     test_inputs.append(image)
     test_targets.append(target)

test_inputs=np.array(test_inputs)
test_targets=np.array(test_targets)
normalised_test_inputs=test_inputs/255
model=keras.models.load_model("Models/best_of_facemask.hdf5")
for i , test in enumerate(normalised_test_inputs):    # for 1 st iteration i will be 0 and test will be normalised_test_inputs[0]
     prediction=model.predict_classes(test.reshape(1,50,50,3))
     cv2.imshow("Image",test)
     print("Target: ",classes[test_targets[i]],"Prediction: ",classes[prediction[0][0]])
     if cv2.waitKey(0)==27:
           break
cv2.destroyAllWindows()
model.evaluate(normalised_test_inputs,test_targets)

