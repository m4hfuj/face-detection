import cv2
from facenet_pytorch import MTCNN
# from deepface import DeepFace




mtcnn = MTCNN()

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if ret:

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces using MTCNN
        boxes, probs = mtcnn.detect(rgb_frame)

        if boxes is not None:
            for box, prob in zip(boxes, probs):
                x, y, w, h = box.astype(int)


                cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
                cv2.putText(frame, str(prob), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)

                # face_image = frame[y:h, x:w]

                cv2.imwrite("Face.jpg", frame)

                ###### put the face image through detection model
                # analysis = DeepFace.analyze(frame, actions = ["age", "gender", "emotion", "race"])

                # cv2.putText(frame, str(analysis), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)

                

        cv2.imshow("Preview", frame)
        cv2.waitKey(1)

