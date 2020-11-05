#!/bin/bash

sleep 5

./camera-feed /dev/video6 left.jpg
./camera-feed /dev/video2 right.jpg

echo -en "\007"
