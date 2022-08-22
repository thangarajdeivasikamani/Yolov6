#pip install youtube-dl==2020.12.2
import cv2
import pafy
from urllib.parse import urlparse,parse_qs

#Ask the user for url input
url = 'https://www.youtube.com/watch?v=XvNsLKd5gsg'

#Getting video id from the url string
url_data = urlparse(url)
query = parse_qs(url_data.query)
id = query["v"][0]
video = 'https://youtu.be/{}'.format(str(id))

#Using the pafy library for youtube videos
urlPafy = pafy.new(video)
videoplay = urlPafy.getbest(preftype="any")

cap = cv2.VideoCapture(videoplay.url)

#Asking the user for video start time and duration in seconds
milliseconds = 1000
start_time = int(input("Enter Start time: "))
end_time = int(input("Enter Length: "))
end_time = start_time + end_time

# Passing the start and end time for CV2
cap.set(cv2.CAP_PROP_POS_MSEC, start_time*milliseconds)
#Will execute till the duration specified by the user
while True and cap.get(cv2.CAP_PROP_POS_MSEC)<=end_time*milliseconds:
        success, img = cap.read()
        cv2.imshow("Image", img)
        cv2.waitKey(1)