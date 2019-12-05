from imageai.Detection import ObjectDetection

#create an instance of ObjectDetection class
detector = ObjectDetection()

model_path = "./models/yolo-tiny.h5"
input_path = "./input/image1.jpg"
output_path = "./output/newimage.jpg"

#use to load model
detector.setModelTypeAsTinyYOLOv3()
#accepts pre-trained model's path
detector.setModelPath(model_path)
#load model
detector.loadModel()

#to detect objects in image
detection = detector.detectObjectsFromImage(input_image=input_path, output_image_path=output_path)

for eachItem in detection:
    print(eachItem["name"], ":", eachItem["percentage_probability"])
