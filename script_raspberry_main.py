!/usr/bin/env python

# Import libraries
import socket
import cv2
import pickle
import struct
import board #for NeoPixels
import neopixel #for NeoPixels

#--------------------------------
# NeoPixels code:
#--------------------------------
# Choose an open pin connected to the Data In of the NeoPixel strip, i.e. board.D18
# NeoPixels must be connected to D10, D12, D18 or D21 to work.
pixel_pin = board.D18

# The number of NeoPixels
num_pixels = 24

# The order of the pixel colors - RGB or GRB. Some NeoPixels have red and green reversed!
# For RGBW NeoPixels, simply change the ORDER to RGBW or GRBW.
ORDER = neopixel.GRB

pixels = neopixel.NeoPixel(
    pixel_pin, num_pixels, brightness=0.2, auto_write=False, pixel_order=ORDER
)
#--------------------------------

#--------------------------------
# Server code:
#--------------------------------
# Socket Create
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host_ip = socket.gethostbyname("raspberrypifabio")
host_ip = '192.168.137.204' # '192.168.137.63'
print('HOST IP:', host_ip)
port = 5550 #8458
socket_address = (host_ip, port)

# Socket Bind
server_socket.bind(socket_address)

# Socket Listen
server_socket.listen(5)
print("LISTENING AT:", socket_address)


# Socket Accept
while True:
    pixels[20] = (255, 0, 0)
    for k in range(2):
        pixels[0+12*k] = (255, 0, 0)
        pixels[1+12*k] = (255, 0, 0)
        pixels[4+12*k] = (0,255,0)
        pixels[5+12*k] = (0,255,0)
        pixels[8+12*k] = (0,0,255)
        pixels[9+12*k] = (0,0,255)
    pixels.show()
    client_socket, addr = server_socket.accept()
   
    
    
    
    
    print('GOT CONNECTION FROM:', addr)
    
    if client_socket:
        vid = cv2.VideoCapture(0)
        while vid.isOpened():
            img, frame = vid.read()
            #frame = imutils.resize(frame, width=320)
            height, width,_ = frame.shape
            resize_perc = 0.5
            height = int(height * resize_perc)
            width = int(width * resize_perc)
            #print(height, width)
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            a = pickle.dumps(frame)
            message = struct.pack("Q", len(a)) + a
            client_socket.sendall(message)



