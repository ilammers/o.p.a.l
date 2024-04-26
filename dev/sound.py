import vlc 
import time 

# this is the original code used to test playing audio over bluetooth
  
# creating vlc media player object 
media_player = vlc.MediaPlayer() 
  
# media object 
media = vlc.Media("/home/pi/Downloads/Djo  I Want Your Video Official Audio.mp3") 
  
# setting media to the media player 
media_player.set_media(media) 
  
# setting video scale 
media_player.video_set_scale(0.6) 
  
# start playing video 
media_player.play() 
  
# wait so the video can be played for 5 seconds 
# irrespective for length of video 
time.sleep(100) 
  
# getting track 
value = media_player.audio_output_device_enum() 
  
# printing value 
print("Audio Output Devices: ") 
print(value) 
