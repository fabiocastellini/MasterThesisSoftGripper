# Per accendere i robot Panda:
- accendere controllore che sta alla base del robot (servono 30 secondi per scaricare i condensatori)
- collegarsi all'indirizzo IP riportato sul robot (es: 192.168.3.1/admin)
- se il gripper (es: Franka Hand) è collegato, impostare la massa aggiuntiva (se necessario) e fare l'homing (inizializzazione apertura/chiusura)

# Per usare i controllori in ROS:
- dal menù desk -> lucchetto per sbloccare frame (giallo -> bianco) 
- dal menù: activate FCI
- sbloccare fungo (bianco -> blu)
- dal computer koga lanciare panda_force_ctrl (roslaunch panda_force_ctrl ... .launch

# Per aprire o chiudere il gripper (Franka Hand):
- rostopic pub /frank_gripper/move/goal frank_gripper/MoveActionGoal "header... 
- inserire i parametri di "speed" e "width"

# ROS:
- il core va lanciato su "james" (ssh altair@James; pw: altair)
- export ROS_HOSTNAME=fabio-pc
- export ROS_MASTER_URI=http://James:11311
- Per modificare i nomi/ip degli host: sudoedit /etc/hosts

# Per usare il sensora forza/coppia ATI Nano 17:
- collegare PoE nella parte sotto del PC e alla presa sotto
- roscore su james con pw altair
- rosrun netft_utils netft_utils_sim
- leggere i topic del sensore (ref: http://wiki.ros.org/netft_utils)
- rostopic echo /netft_data