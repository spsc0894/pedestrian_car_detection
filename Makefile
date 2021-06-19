install:
	docker build -t detection_app .
execute:
	xhost +
	docker run -v /tmp/.X11-unix:/tmp/.X11-unix --env="DISPLAY" detection_app:latest

