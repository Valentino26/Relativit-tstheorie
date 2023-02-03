from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from scipy.optimize import fsolve
import math

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

line, = ax.plot3D([],[],[],'.')
line1, = ax.plot3D([],[],[])

# Setting the axes properties
ax.set_xlim3d([-20.0, 20.0])
ax.set_xlabel('X')

ax.set_ylim3d([-20.0, 20.0])
ax.set_ylabel('Y')

ax.set_zlim3d([-20.0, 20.0])
ax.set_zlabel('Z')



def LorentzBoost(v_vector,co_vector): #boosts the vector co_vector with the speed v_vector

    v_x = v_vector[0]
    v_y = v_vector[1]
    v_z = v_vector[2]
    

    v = np.sqrt(v_x**2 + v_y**2 + v_z**2)

    c = 1 #c := 100% the speed of light

    gamma = 1/np.sqrt(1-(v**2/c**2))

    B = np.array([[gamma, ((-gamma*v_x)/c), ((-gamma*v_y)/c), ((-gamma*v_z)/c)],
                [((-gamma*v_x)/c), (1 + ((gamma-1)*(((v_x**2))/v**2))), ((gamma-1)*((v_x*v_y)/v**2)), ((gamma-1)*((v_x*v_z)/v**2))],
                [((-gamma*v_y)/c), ((gamma-1)*((v_y*v_x)/v**2)), (1 + ((gamma-1)*(((v_y**2))/v**2))) , ((gamma-1)*((v_y*v_z)/v**2))],
                [((-gamma*v_z)/c), ((gamma-1)*((v_z*v_x)/v**2)), ((gamma-1)*((v_z*v_y)/v**2)), (1 + ((gamma-1)*(((v_z**2))/v**2)))]])

    return np.matmul(B, co_vector)


def functiontosolve(lam,tau,x,y,v_vector):

    x = np.array([lam[0],x[0],x[1],x[2]])
    o = np.array([tau,y[0],y[1],y[2]])
    xP = LorentzBoost(v_vector,x)

    s = np.subtract(xP,o)

    d = np.linalg.norm(s[1:])

    return(d+xP[0]-tau)

def Transform(x,y,v_vector,tau):

    lam = fsolve(functiontosolve,x0=tau,args=(tau,x,y,v_vector))

    x = np.array([lam[0],x[0],x[1],x[2]])
    xP = LorentzBoost(v_vector,x)


    return xP

def get_sphere_coordinates(radius, center_x, center_y, center_z):

    x_coordinates = []
    y_coordinates = []
    z_coordinates = []

    

    for theta in np.arange(0, 2 * math.pi, 0.7):
        for phi in np.arange(0, 1 * math.pi, 0.6):
            x = center_x + radius * math.sin(phi) * math.cos(theta)
            y = center_y + radius * math.sin(phi) * math.sin(theta)
            z = center_z + radius * math.cos(phi)

            x_coordinates.append(x)
            y_coordinates.append(y)
            z_coordinates.append(z)

    return x_coordinates,y_coordinates,z_coordinates

sphere_coordinates = get_sphere_coordinates(3, 1, 3, 4)
    ##print(sphere_coordinates)

def positionTransform(array_x, array_y,array_z,o,v,ntau):
    pnposition = []
    
    def get_min_array(array1, array2, array3):

        len1 = len(array1)
        len2 = len(array2)
        len3 = len(array3)

        if len1 <= len2 and len1 <= len3:
            return array1
        elif len2 <= len1 and len2 <= len3:
            return array2
        else:
            return array3

    for i in range(0, len(array_x)):
        pnposition.append(Transform(np.array([array_x[i],array_y[i],array_z[i]]),o,v,ntau)[1:])

    return pnposition




def update(ntau):
    data = []

    o = np.array([0,10,0])

    v_rs = np.array([0,0.95,0])

    pnposition = positionTransform(sphere_coordinates[0],sphere_coordinates[1],sphere_coordinates[2],o,v_rs,ntau)

    data.append(pnposition)
    

    x = []
    y = []
    z = []

    for i in range(0,len(data)):
        for n in range(0,len(data[0])):
            x.append(data[i][n][0])
            y.append(data[i][n][1])
            z.append(data[i][n][2])
    
    line.set_data(x[:756],y[:756])
    line.set_3d_properties(z[:756])
    

    return line,

ani = animation.FuncAnimation(fig, update, frames= 100, interval=100, blit=False)
plt.show()