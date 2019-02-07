import numpy as np

theta = np.pi/2
tee = 0.5
teh = 0
e_sigma = 40
U = 50	
e_delta = 0
bL = 6.5
bR = 5.5

c = np.cos(theta/2)
s = np.cos(theta/2)

delta_up_down = (teh*c)**2/(-(bL-bR)+e_sigma)+(tee*c)**2*(1/(-(bL-bR)-(U-e_delta))+1/(-(bL-bR)-(U+e_delta)))
delta_down_up = (teh*c)**2/(bL-bR+e_sigma)+(tee*c)**2*(1/(bL-bR-(U-e_delta))+1/(bL-bR-(U+e_delta)))
delta_t_p = (teh*s)**2/(-(bL+bR)+e_sigma)+(tee*s)**2*(1/(-(bL+bR)-(U-e_delta)) +1/(-(bL+bR)-(U+e_delta)))
delta_t_m = (teh*s)**2/(bL+bR+e_sigma)+(tee*s)**2*(1/(bL+bR-(U-e_delta))+1/(bL+bR-(U+e_delta)))

print(delta_up_down)
print(delta_down_up)
print(delta_t_p)
print(delta_t_m)
shift_L_tp = delta_down_up - delta_t_p
shift_L_tm = delta_up_down - delta_t_m
shift_R_tp = delta_up_down - delta_t_p
shift_R_tm = delta_down_up - delta_t_m

print("Shift cavite L pour l'état T+ : "+str(shift_L_tp))
print("Shift cavite R pour l'état T+ : "+str(shift_R_tp))
print("Shift cavite L pour l'état T- : "+str(shift_L_tm))
print("Shift cavite R pour l'état T- : "+str(shift_R_tm))

real = [(50,1, 0.0117,3672), (150,1,0.0077 ,3520),(250,1, 0.0074, 3492), (250, 0.5, 0.0072, 3483), (100,0.5,0.0075,3551), (50,0.5, 0.0086,3496)]