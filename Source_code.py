import gym
import pix_main_arena
import time
import pybullet as p
import pybullet_data
import cv2
import os
import numpy as np
import math
import cv2.aruco as aruco
#algo-------------------------------------------------------------------
from collections import deque, namedtuple

global kmp
flag=0
# we'll use infinity as a default distance to nodes.
inf = float('inf')
Edge = namedtuple('Edge', 'start, end, cost')


def make_edge(start, end, cost=1):
  return Edge(start, end, cost)


class Graph:
    def __init__(self, edges):
        # let's check that the data is right
        wrong_edges = [i for i in edges if len(i) not in [2, 3]]
        if wrong_edges:
            raise ValueError('Wrong edges data: {}'.format(wrong_edges))

        self.edges = [make_edge(*edge) for edge in edges]

    @property
    def vertices(self):
        return set(
            sum(
                ([edge.start, edge.end] for edge in self.edges), []
            )
        )

    def get_node_pairs(self, n1, n2, both_ends=True):
        if both_ends:
            node_pairs = [[n1, n2], [n2, n1]]
        else:
            node_pairs = [[n1, n2]]
        return node_pairs

    def remove_edge(self, n1, n2, both_ends=True):
        node_pairs = self.get_node_pairs(n1, n2, both_ends)
        edges = self.edges[:]
        for edge in edges:
            if [edge.start, edge.end] in node_pairs:
                self.edges.remove(edge)

    def add_edge(self, n1, n2, cost=1, both_ends=True):
        node_pairs = self.get_node_pairs(n1, n2, both_ends)
        for edge in self.edges:
            if [edge.start, edge.end] in node_pairs:
                return ValueError('Edge {} {} already exists'.format(n1, n2))

        self.edges.append(Edge(start=n1, end=n2, cost=cost))
        if both_ends:
            self.edges.append(Edge(start=n2, end=n1, cost=cost))

    @property
    def neighbours(self):
        neighbours = {vertex: set() for vertex in self.vertices}
        for edge in self.edges:
            neighbours[edge.start].add((edge.end, edge.cost))

        return neighbours

    def dijkstra(self, source, dest):
        assert source in self.vertices, 'Such source node doesn\'t exist'
        distances = {vertex: inf for vertex in self.vertices}
        previous_vertices = {
            vertex: None for vertex in self.vertices
        }
        distances[source] = 0
        vertices = self.vertices.copy()

        while vertices:
            current_vertex = min(
                vertices, key=lambda vertex: distances[vertex])
            vertices.remove(current_vertex)
            if distances[current_vertex] == inf:
                break
            for neighbour, cost in self.neighbours[current_vertex]:
                alternative_route = distances[current_vertex] + cost
                if alternative_route < distances[neighbour]:
                    distances[neighbour] = alternative_route
                    previous_vertices[neighbour] = current_vertex

        path, current_vertex = deque(), dest
        while previous_vertices[current_vertex] is not None:
            path.appendleft(current_vertex)
            current_vertex = previous_vertices[current_vertex]
        if path:
            path.appendleft(current_vertex)
        return path

#----------------------------------------------------------------------------------------------------------------
def list_of_tupples(graph,direction,ep):
    endp=ep
    lst=[]
    for row in range(0,12):
        for column in range(0,12):
            #starting from 0,0
            #1,2,3,...= row*12+column+1
            #-3 for up
            #-4 for down
            #-5 for left
            #-6 for right
            ##ARRAY INDEX OUT OF BOND ERREO AAYEGA
            #ABHI YE CORRECT KRNA HAI
            sln=row*12+column+1

            if column !=11:
                tpl=(sln,sln+1,graph[row][column+1][2])
                if direction[row][column+1]!=-5:#-5 for left
                     lst.append(tpl)

            if column !=0:
                tpl=(sln,sln-1,graph[row][column-1][2])
                if direction[row][column-1]!=-6:#-6 for right
                     lst.append(tpl)

            if row !=0:
                tpl=(sln,sln-12,graph[row-1][column][2])
                if direction[row-1][column]!=-4:#-4 for down
                     lst.append(tpl)

            if row !=11:
                tpl=(sln,sln+12,graph[row+1][column][2])
                if direction[row+1][column]!=-3: #-3 for up
                    lst.append(tpl)

    #print(lst)
    gp = Graph(lst)
    print(gp.dijkstra(144,endp))
    global kmp
    kmp=gp.dijkstra(144,endp)
    
def top_x(approx):
    min_y=approx[0][0][1]
    top_x=approx[0][0][0]
    for i in range(0,len(approx)):
        for j in range(0,len(approx[i])):
            for k in range(0,len(approx[i][j])):
                if approx[i][j][1]<min_y:
                    min_y=approx[i][j][1]
                    top_x=approx[i][j][0]
    return top_x,min_y
global k
k=0
def triangle_at(r_img,h,w):
    raw_img=r_img
    #create empty list
    direction=[]
    for row in range(0,12):
        row_ele=[]
        row_ele.clear()
        for column in range(0,12):
            row_ele.append(0)
        direction.append(row_ele)
   

    comp_img=cv2.resize(raw_img,(0,0),fx=0.5,fy=0.5)
    comp_img=cv2.GaussianBlur(comp_img,(5,5),0) 
    hsv_img=cv2.cvtColor(comp_img,cv2.COLOR_BGR2HSV)
    ldb=np.array([115,200,200])
    udb=np.array([125,255,255])
    dBlueMask=cv2.inRange(hsv_img,ldb,udb)
    finaldBlueMask=cv2.bitwise_and(comp_img,comp_img,mask=dBlueMask)

    contours, _= cv2.findContours(dBlueMask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    print('Number of contours found : ', len(contours))
    for i in range(0,len(contours)):
        approx=cv2.approxPolyDP(contours[i],0.05*cv2.arcLength(contours[i],True),True)
        cv2.drawContours(comp_img,approx,-1,(255,255,255),2)
        cv2.imshow("contors",comp_img)
        x=approx.ravel()[0]
        y=approx.ravel()[1]
        if len(approx)==3:
            tx,my=top_x(approx)
            cv2.putText(comp_img,"Triangle",(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255))
            M = cv2.moments(contours[i])
            #print(approx)
            if(M['m00'] !=0):
                cx = int(M['m10']/M['m00']) #x cordinate of centroid
                cy = int(M['m01']/M['m00']) #y cordinate of centroid
                xi=int(cx/w)
                yi=int(cy/h)
                #-3 for up
                #-4 for down
                #-5 for left
                #-6 for right
                vec=complex(cx-tx,cy-my)
                print("tx",tx)
                global k
                direction[yi][xi]=k
                k=k+1
                
                ang=np.angle(vec,deg=True)
                print("angle",ang)
                if ang>50 and ang<70 :
                  direction[yi][xi]=-6
                if ang<50 and ang>20 :
                  direction[yi][xi]=-4
                if ang>80 and ang<110 :
                  direction[yi][xi]=-3
                if ang>110 and ang<140 :
                  direction[yi][xi]=-5
                """if ang<110 and ang>80 :#right of left
                        direction[yi][xi]=-3
                elif ang>-110 and ang<-80:#right of left
                        direction[yi][xi]=-4
                elif ang<-20 and ang>-40:#right of left
                        direction[yi][xi]=-6
                elif ang>-160 and ang<-140 :#right of left
                        direction[yi][xi]=-5"""
    

              
                #print(cx,cy)
    #print(direction)
    for row in range(0,12):
        for column in range(0,12):
            print(direction[row][column],end=" ")
        print()
    return direction


def first(img):
    #read image
    #raw_img=cv2.imread('A:\\technex\\pixilate\\map111.jpg')
    raw_img=img
    comp_img=cv2.GaussianBlur(raw_img,(5,5),0) 
    comp_img=cv2.resize(comp_img,(0,0),fx=0.5,fy=0.5)
    hsv_img=cv2.cvtColor(comp_img,cv2.COLOR_BGR2HSV)
    INF=99999 #VALUE WILL BE CHANGED
    h=int(comp_img.shape[0]/12)
    w=int(comp_img.shape[1]/12)
    #CREATING ARRAY
    graph=[]
    bc=1        #box count
    for row in range(0,12):
        row_ele=[]
        row_ele.clear()
        for column in range(0,12):
            row_ele.append([bc,INF,INF,0])
            bc+=1
        graph.append(row_ele)
      
    #CENTROID
    #red
    lr=np.array([0,150,50])
    ur= np.array([10,255,255])
    red1=cv2.inRange(hsv_img,lr,ur)

    ulr=np.array([170,150,50])
    uur= np.array([180,255,255])
    red2=cv2.inRange(hsv_img,ulr,uur)

    redMask=red1+red2

    finalred=cv2.bitwise_and(comp_img,comp_img,mask=redMask)
    #cv2.imshow("red Mask",finalred)
    contours, _= cv2.findContours(redMask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    print('Number of red contours found : ', len(contours))
    #finding cordinates of centroid of contord detected
    list_of_red=[]
    #Append value to array
    for i in range(0,len(contours)):
        M = cv2.moments(contours[i])
        if (M["m00"]!=0):
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            #print("Centroid of yellow countor :",cx,", ",cy)
            list_of_red.append([cx,cy])
    #ASSSIGNING VALUE OF RED WEIGHT TO ARRAY
    for i in range(0,len(list_of_red)):
        x=int(list_of_red[i][1]/w)
        y=int(list_of_red[i][0]/h)
        graph[x][y][2]=4
#-------------------------------------------------------------------------------------------------
    #yellow
    ly=np.array([25,150,50])
    uy=np.array([45,255,255])
    yelMask=cv2.inRange(hsv_img,ly,uy)
    finalyel=cv2.bitwise_and(comp_img,comp_img,mask=yelMask)
    #cv2.imshow("yellow Mask",finalyel)
    contours, _= cv2.findContours(yelMask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    print('Number of Yellowcontours found : ', len(contours))
    #finding cordinates of centroid of contor detected
    list_of_ylw=[]
    #Append value to array
    for i in range(0,len(contours)):
        M = cv2.moments(contours[i])
        if (M["m00"]!=0):
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            #print("Centroid of yellow countor :",cx,", ",cy)
            list_of_ylw.append([cx,cy])
    #ASSSIGNING VALUE OF RED WEIGHT TO ARRAY
    for i in range(0,len(list_of_ylw)):
        x=int(list_of_ylw[i][1]/w)
        y=int(list_of_ylw[i][0]/h)
        graph[x][y][2]=3
#------------------------------------------------------------------------------------------------------
    #green
    lg=np.array([60,200,150])
    ug=np.array([80,255,255])
    greMask=cv2.inRange(hsv_img,lg,ug)
    finalgre=cv2.bitwise_and(comp_img,comp_img,mask=greMask)
    #cv2.imshow("Green Mask",finalgre)
    contours, _= cv2.findContours(greMask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    print('Number of Green contours found : ', len(contours))
    #finding cordinates of centroid of contord detected
    list_of_green=[]
    #Append value to array
    for i in range(0,len(contours)):
        M = cv2.moments(contours[i])
        if (M["m00"]!=0):
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            #print("Centroid of yellow countor :",cx,", ",cy)
            list_of_green.append([cx,cy])
    #ASSSIGNING VALUE OF RED WEIGHT TO ARRAY
    for i in range(0,len(list_of_green)):
        x=int(list_of_green[i][1]/w)
        y=int(list_of_green[i][0]/h)
        graph[x][y][2]=2
#----------------------------------------------------------------------------------------
    #white
    luw=np.array([0,0,220])
    uuw=np.array([0,0,255])
    whMask=cv2.inRange(hsv_img,luw,uuw)
    finalwh=cv2.bitwise_and(comp_img,comp_img,mask=whMask)
    #cv2.imshow("White Mask",finalwh)
    contours, _= cv2.findContours(whMask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    print('Number of White contours found : ', len(contours))
    #finding cordinates of centroid of contord detected
    list_of_white=[]
    #Append value to array
    for i in range(0,len(contours)):
        M = cv2.moments(contours[i])
        if (M["m00"]!=0):
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            #print("Centroid of yellow countor :",cx,", ",cy)
            list_of_white.append([cx,cy])
    #ASSSIGNING VALUE OF RED WEIGHT TO ARRAY
    for i in range(0,len(list_of_white)):
        x=int(list_of_white[i][1]/w)
        y=int(list_of_white[i][0]/h)
        graph[x][y][2]=1
#------------------------------------------------------------------
    #pink and light blue
    lp=np.array([145,95,190])
    up=np.array([155,255,255])
    pinkMask=cv2.inRange(hsv_img,lp,up)

    llb=np.array([85,190,190])
    ulb=np.array([95,255,255])
    lblueMask=cv2.inRange(hsv_img,llb,ulb)

    fMask=pinkMask+lblueMask

    finalmask=cv2.bitwise_and(comp_img,comp_img,mask=fMask)
    #cv2.imshow("Green Mask",finalgre)
    contours, _= cv2.findContours(fMask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    print('Number of Green contours found : ', len(contours))
    #finding cordinates of centroid of contord detected
    list_of_hos_pat=[]
    #Append value to array
    for i in range(0,len(contours)):
        M = cv2.moments(contours[i])
        if (M["m00"]!=0):
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            #print("Centroid of yellow countor :",cx,", ",cy)
            list_of_hos_pat.append([cx,cy])
    #ASSSIGNING VALUE OF RED WEIGHT TO ARRAY
    for i in range(0,len(list_of_hos_pat)):
        x=int(list_of_hos_pat[i][1]/w)
        y=int(list_of_hos_pat[i][0]/h)
        graph[x][y][2]=1
#-------------------------------------------------------------------
    #[sl.no,distance,waight,parent]: order of element
    direction=triangle_at(raw_img,h,w)           #this will print position of triangle##############

    #for decoding starting and end position
    lp=np.array([145,95,190])
    up=np.array([155,255,255])
    pinkMask=cv2.inRange(hsv_img,lp,up)

    finalmask=cv2.bitwise_and(comp_img,comp_img,mask=pinkMask)
    #cv2.imshow("Green Mask",finalgre)
    contours, _= cv2.findContours(pinkMask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    print('Number of Pink contours found : ', len(contours))
    #finding cordinates of centroid of contord detected
    list_of_pink=[]
    #Append value to array
    for i in range(0,len(contours)):
        M = cv2.moments(contours[i])
        if (M["m00"]!=0):
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            #print("Centroid of yellow countor :",cx,", ",cy)
            list_of_pink.append([cx,cy])
    #ASSSIGNING VALUE OF RED WEIGHT TO ARRAY
    k=0
    x=int(list_of_pink[k][1]/w)
    y=int(list_of_pink[k][0]/h)

    end=12*x+y+1
    
    list_of_tupples(graph,direction,end)

    for r in range(0,12):
        for c in range(0,12):
            print(graph[r][c][2],end=' ')
        print()
    
def vandc( num):
         img = env.camera_feed()
         cv2.imshow("img", img)
         cv2.waitKey(10)
       
         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
         aruco_dict=aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
         parameters=aruco.DetectorParameters_create()
         corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters= parameters)
         "corners=np.squeeze(corners)"
         "ids=np.squeeze(ids)"
         "print('ID=',ids)"
         "print(corners)"
         "print(img.shape)"
         for i, corner in zip(ids, corners):
             print('ID: {}; Corners: {}'.format(i, corner))
         x1=corners[0][0][0][0]
         y1=corners[0][0][0][1]
         x2=corners[0][0][3][0]
         y2=corners[0][0][3][1]
         print("cornnner",corners[0][0][1][0])
         cbx=(x1+corners[0][0][1][0]+corners[0][0][2][0]+corners[0][0][3][0])/4.0
         cby=(y1+corners[0][0][1][1]+corners[0][0][2][1]+corners[0][0][3][1])/4.0
         print("cbx",cbx)
         print("cby",cby)
         v1=complex(x1-x2,y1-y2)
         d=dict();
         d['c']=cbx
         d['cy']=cby
         d['v1']=v1
         return d
def forward ():
                global flag
                print("hellofor")
                while True:
                  if (flag==1):
                     break
                  f=vandc(6)
                  v2=complex(x4-f['c'],y4-f['cy'])
                  ang1=np.angle(f['v1'],deg=True)
                  ang2=np.angle(v2,deg=True)
                  print("ang1",ang1)
                  print("ang1",ang2)
                  """if( ang1>0 and ang2<0):
                   ang=360-(ang1-ang2)
                  elif( ang1<0 and ang2>0):
                   ang=(-ang1+ang2)-360
                  elif(ang1<ang2 and ang1<0 and ang2<0):
                   ang=ang1-ang2
                  elif(ang1>ang2 and ang1<0 and ang2<0):
                   ang=ang1-ang2
                  elif(ang1<ang2 and ang1>0 and ang2>0):
                   ang=ang1-ang2
                  elif(ang1>ang2 and ang1<0 and ang2<0):
                   ang=ang1-ang2
                  else:
                     print("cvhvcivsilgvsi")"""
                  #ang1=abs(ang1)
                  #ang2=abs(ang2)
                  #ang= ang1-ang2
                  ang=np.angle(v2/f['v1'])
                  ang=(ang*180)/3.14
                  
                  app=5
                  dis=(x4-f['c'])*(x4-f['c'])+(y4-f['cy'])*(y4-f['cy'])
                  dis=math.sqrt(dis)
                  print ("dis",dis)
                  thres=5
                  
                  if (dis>thres):
                      
                
                      env.move_husky(0.5, 0.5, 0.5, 0.5)
                      for i in range(200):
                         p.stepSimulation()
                      if  ((ang<-app or ang>app) and dis>thres):
                       print("turn")
                       turn()
                  elif(dis<=thres):
                        print ("reached")
                        flag=1
                        break
                       
                
def turn():
                global flag
                while True:
                 if(flag==1):
                     break
                 print ("hellot")
                 f=vandc(6)
                 thres=5
                 dis=(x4-f['c'])*(x4-f['c'])+(y4-f['cy'])*(y4-f['cy'])
                 dis=math.sqrt(dis)
                 print ("dis",dis)
                 v2=complex(x4-f['c'],y4-f['cy'])
                 ang1=np.angle(f['v1'])
                 print("ang1",ang1)
                 
                 ang=np.angle(v2/f['v1'])
                 ang=(ang*180)/3.14
                 
                 
                 
                 """if( ang1>0 and ang2<0):
                   ang=360-(ang1-ang2)
                 elif( ang1<0 and ang2>0):
                   ang=(-ang1+ang2)-360
                 elif(ang1<ang2 and ang1<0 and ang2<0):
                   ang=ang1-ang2
                 elif(ang1>ang2 and ang1<0 and ang2<0):
                   ang=ang1-ang2
                 elif(ang1<ang2 and ang1>0 and ang2>0):
                   ang=ang1-ang2
                 elif(ang1>ang2 and ang1<0 and ang2<0):
                   ang=ang1-ang2
                 else:
                     print("cvhvcivsilgvsi")"""
    
                 #ang= abs(ang1)-abs(ang2)
                 
                 #print("ang2",ang2)
                 print (ang)
                 app=5
                 if (ang>-app and ang<app and dis>thres):
                        print("i sm ")
                        env.move_husky(0, 0, 0, 0)
                        for i in range (5): 
                           p.stepSimulation()  
                        forward()
                 elif (ang<-app and dis>thres):
                       print("i sm2 ")
                       env.move_husky(-0.3,0.3, -0.3,0.3)
                       
                       
                       for i in range (100):
                            p.stepSimulation()
                       env.move_husky(0, 0, 0, 0)
                       for i in range (10): 
                           p.stepSimulation()  
                 elif (ang>app and dis>thres):
                      print("i sm3 ")
                      env.move_husky(0.3, -0.3, 0.3, -0.3)
                      t=1
                      if(ang>40 or ang <-40):
                        t=200
                      else:
                           t=100
                      for i in range (t):
                            p.stepSimulation()
                      env.move_husky(0, 0, 0, 0)
                      for i in range (10): 
                           p.stepSimulation()
                 elif(dis<=thres ):
                      flag=1
                
def propogate():
             global flag
             if(flag==0):
                 forward()
             if(flag==1):
                for i in range (20):
                 print("reached")

    
if __name__=="__main__":
    parent_path = os.path.dirname(os.getcwd())
    os.chdir(parent_path)
    env = gym.make("pix_main_arena-v0")
    time.sleep(3)
    env.remove_car()
    time.sleep(1)
    image = env.camera_feed()
    image=image[40:685,40:685]
    first(image)
    time.sleep(1)
    env.respawn_car()
    time.sleep(1)
    global kmp
    path=kmp
    for i in range (len(path)):
      if(path[i]%12!=0):
            row= math.floor(path[i]/12+1)
            path[i]=13+path[i]+2*row
      else:
        row= math.floor(path[i]/12+1)
        path[i]=13+path[i]+2*row-2
    print(path)
    print ("pathkgnwrgne gnegnenkgepgne jge")
    cv2.imshow("img", image)
    cv2.waitKey(10)
   
    p.stepSimulation()
        
    for j in range(1,len(path)):
            print("hello node")
            pix=720
            t=14
            node=path[j]
            print ("node",node)
            y4=((int)(node/t)+1)*(pix/t)-pix/(t*2)+20
            if (node%t==0):
                x4=12*(pix/t)-pix/(t*2)+20
            else:
                x4=(node%t)*(pix/t)-pix/(t*2)+20
            app=10
            print("x4",x4)
            print("y4",y4)
            
            propogate()
            flag=0
            for i in range(10):
              print("i am in node",node)
    print("stop")
    env.move_husky(0, 0, 0, 0)
    p.stepSimulation()
                    
