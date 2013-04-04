import numpy as np
from Tkinter import *

import random
import argparse
import sys


def isblockinbig(j, big,p):
	#print '---', len(big)
	for k in range(0,len(big)):
		
		if j == big[k][p]:
			#print j, big
			return True
		
	return False
			
			
			

def gen_grid(width, bw, b):
	''' 
	generate a grid with bw columns and b rows. bw/2 columns are left of centre and b/2 rows are above the centre.
	the rows and columns have an irregular width
	'''

	posx = np.floor(1.0*width/bw)*np.ones(bw+1)
	posy = np.floor(1.0*width/b)*np.ones(b+1)
	posx[0] = 0
	posy[0] = 0
		
	randwn = np.zeros(bw+1)
	randvn = np.zeros(b+1)
	randwn[1:width-bw*np.floor(width/bw)+1] =1
	randvn[1:width-b*np.floor(width/b)+1] =1
	
	random.shuffle(randwn[1:bw+1])
	random.shuffle(randvn[1:b+1])
	
	# randomly lengthen or shorten the x and y dimensions of grid blocks by 0 to 4 pixels
	# the dimensions of the blocks are restricted such that the total width and height add up to width
	# the dimensions of the blocks are such that the half the blocks are left of width/2 and half are above width/2
	
	for i in range(4):
		n = bw/8
		m = bw/2 
		randwn[1:n+1] = randwn[1:n+1] +1
		randwn[n+2:2*n+2] = randwn[n+2:2*n+2] -1
		randwn[m:m+n] = randwn[m:m+n] +1
		randwn[m+n+1:m+2*n+1] = randwn[m+n+1:m+2*n+1] -1	
		
		n = b/8
		m = b/2
		randvn[1:n+1] = randvn[1:n+1] +1
		randvn[n+2:2*n+2] = randvn[n+2:2*n+2] -1
		randvn[m:m+n] = randvn[m:m+n] +1
		randvn[m+n+1:m+2*n+1] = randvn[m+n+1:m+2*n+1] -1	
		
		random.shuffle(randwn[1:np.floor((bw+1)/2.0)])
		random.shuffle(randvn[1:np.floor((b+1)/2.0)])	
		
		random.shuffle(randwn[np.ceil((bw+1)/2.0):bw+1])
		random.shuffle(randvn[np.ceil((b+1)/2.0):b+1])	
			
	
	for i in range(1,bw+1):
		posx[i] = posx[i]+posx[i-1] + randwn[i]
	for i in range(1,b+1):
		posy[i] = posy[i]+ posy[i-1] + randvn[i]	
		
	assert posx[bw] == width and posy[b] == width
	
	return posx, posy






def colour_grid(width, bw, b):
	'''
	fill the blocks with a random colour which is different from the colour of the blocks left and above
	'''
	cplot = np.zeros((bw,b), dtype=int)
	
	for i in range(0,bw):
		for j in range(0,b):
		
			allowedcolours = list(set(range(len(colour)-2)) - set([cplot[i-1,j], cplot[i,j-1]]))
			cplot[i,j] = int(random.choice(allowedcolours))
			
			'''
			gen = True
			while gen == True:
				cplot[i,j] = int(random.randint(0, len(colour)-2))
				gen = True
				if i < 1 or j < 1 or (cplot[i,j] != cplot[i-1,j] and cplot[i,j] != cplot[i,j-1]):
					gen = False
					if i > 2 and j > 2 and (cplot[i,j] == cplot[i-2,j] or cplot[i,j] == cplot[i,j-2]):
						gen = True
			'''		
	
	return cplot




def plot_canvas(posx, posy, cplot, occupied, colour, xextragrid, yextragrid, drawtriangles):	
	bw = len(posx)-1
	b = len(posy)-1
	
	# turn 1x1 blocks black if they are light coloured, lie next to larger blocks and don't already border a black block		
	for i in range(1,bw-1):
		for j in range(1,b-1):
			#print i,j, bw, b
			if occupied[i,j] ==0 and (occupied[i,j+1] >=1 or occupied[i,j-1] >=1 or occupied[i+1,j] >=1 or occupied[i-1,j] >=1):
				if cplot[i,j] < 3 and cplot[i-1,j] != len(colour) -1  and cplot[i,j-1] != len(colour) -1 :
					cplot[i,j] = len(colour) -1 
				neighbours = [cplot[i-1,j], cplot[i+1,j],cplot[i,j-1] ,cplot[i,j+1] ]
				if 0 in neighbours or 1 in neighbours:
					neighbours = neighbours + [0,1]
					
				if cplot[i,j] in neighbours:	
					#print list(set(range(5,len(colour)-1))-set(neighbours)), neighbours
					cplot[i,j] = random.choice(list(set(range(4,len(colour)))-set(neighbours)))		


	#plot twice to reduce aliasing in ps format
	for iter in [0,1]:
		if iter == 0:
			m1 = b
			m2 = bw
		else:
			m1 = bw
			m2 = b
			
		for l in range(0,m1):
			for k in range(0,m2):
				if iter == 0:
					j = l
					i = k
					aax = 0
					aay = 0
				else:
					j = k
					i = l
					aax = 0
					aay = -1


				add = 0
				addy = 0
				if occupied[i,j]==0 :
					addy =  0# np.round(posx[len(posx)-1]/150.0)
				
				if occupied[i,j]==4:
					add =  np.round(posx[len(posx)-1]/300.0)
					#w.create_rectangle(posx[i]-xextragrid[i,j]-int(random.randrange(0,int(width/bw/2)))+1-add-2, posy[j]-yextragrid[i,j]+1-addy-2-aay, posx[i+1]+aax, posy[j+1]+aay, fill=colour[cplot[i,j]], outline=colour[cplot[i,j]])
					w.create_rectangle(posx[i]-xextragrid[i,j]-add, posy[j]-yextragrid[i,j]-add, posx[i+1]+aax, posy[j+1]+aay, fill=colour[cplot[i,j]], outline=colour[cplot[i,j]])
				
				#elif i > 1 and occupied[i-1,j] ==0 and occupied[i,j]==0:
				#	w.create_rectangle(posx[i]-xextragrid[i,j]-int(random.randrange(0,int(width/bw/2)))+1, posy[j]-yextragrid[i,j]+1-addy, posx[i+1], posy[j+1], fill=colour[cplot[i,j]], outline=colour[cplot[i,j]])
				
				elif occupied[i,j] != 3:
					#w.create_rectangle(posx[i]-xextragrid[i,j]+1-add, posy[j]-yextragrid[i,j]+1-addy, posx[i+1]+aax, posy[j+1]+aay, fill=colour[cplot[i,j]], outline=colour[cplot[i,j]])
					w.create_rectangle(posx[i]-xextragrid[i,j], posy[j]-yextragrid[i,j], posx[i+1]+aax, posy[j+1]+aay, fill=colour[cplot[i,j]], outline=colour[cplot[i,j]])
				else:
					w.create_rectangle(posx[i], posy[j], posx[i+1]+aax, posy[j+1]+aay, fill=colour[cplot[i,j]], outline=colour[cplot[i,j]])


			
				# insert a narrow vertical colour block into 1x1 blocks bordered on the left and right by 1x1 blocks
				if i > 1 and i < bw-1 and occupied[i-1,j] ==0 and occupied[i-1,j] ==0 and occupied[i+1,j] ==0 and (posx[i+1]-posx[i])/2.0>7:
					c = random.randrange(3,len(colour)-1)
					#w.create_rectangle(posx[i]+(posx[i+1]-posx[i])/2.0+1-add, posy[j]-yextragrid[i,j]+1-addy-aay, posx[i+1]+aax, posy[j+1]+aay, fill=colour[c], outline=colour[c])
					w.create_rectangle(posx[i]+(posx[i+1]-posx[i])/2.0+1, posy[j]-yextragrid[i,j], posx[i+1]+aax, posy[j+1]+aay, fill=colour[c], outline=colour[c])
	
	if drawtriangles:
		# draw triangular borders
		w.create_polygon(0,0, 0,width/2, width/2,0, fill=colour[1], outline=colour[1])
		w.create_polygon(0,width, 0,width/2, width/2,width, fill=colour[1])
		w.create_polygon(width,0, width,width/2, width/2,0, fill=colour[1],outline=colour[1])
		w.create_polygon(width,width,  width/2, width, width,width/2, fill=colour[1],outline=colour[1])
	



def get_neighbours(cplot,gridh, gridv, l, i,j, extrax, extray, occupied):
	''' l = gridlevel
	'''
	neighbours = []

	if sum(occupied[gridh[l][i]-2, gridv[l][j]+1:gridv[l][j+1]]) != 0:
		for q in range(0, -gridv[l][j]-1+gridv[l][j+1]):
			if i >= 1 and extrax == 0: #and (cplot[gridh[l][i-1]+1, gridv[l][j]+1] != 0 ):
				neighbours.append(cplot[gridh[l][i]-2, gridv[l][j]+q+1])

	if sum(occupied[gridh[l][i]+1:gridh[l][i+1], gridv[l][j]-2]) !=0:	
		for q in range(0, -gridh[l][i]-1+gridh[l][i+1]):
			if (j >= 1 and extray==0) or j==2 : # and (cplot[gridh[l][i]+1, gridv[l][j-1]+1] != 0):	
				neighbours.append(cplot[gridh[l][i]+q+1, gridv[l][j]-1])
				#neighbours.append(cplot[gridh[l][i]+q, gridv[l][j-1]+2])
	
	#white and ivory should not lie next to each other
	if 1 in neighbours:
		neighbours.append(0)
	if 0 in neighbours:
		neighbours.append(1)	
	
	#light grey and ivory should not lie next to each other
	if 1 in neighbours:
		neighbours.append(2)
	if 2 in neighbours:
		neighbours.append(1)
		
	return list(set(neighbours))	
	
	
	
	
def gen_coarsegrids(bw,b,scale):

	gridv = [[0],[0]] # np.zeros((N,len(posx)), dtype=int)
	gridh = [[0],[0]] # np.zeros((N,len(posx)), dtype=int)
	
	x = 0
	while x < bw-1:
		i = int(random.gauss(1.0*b/scale, scale/67.0)) #12 #5.5
		x = x+i
		if x < bw-1 and i > 1:
			gridh[0].append(x)
			gridh[1].append(int(x-(i+1)/2))
			gridh[1].append(x)
		elif x >= bw-1 and i > 1:	
			gridh[0].append(bw)
			gridh[1].append(bw-(i+1)/2)
			gridh[1].append(bw)
			
	y = 0
	while y < b-1:
		i = int(random.gauss(1.0*b/scale, scale/67.0)) #11 #5.0
		y = y+i
		if y < b-1 and i > 1:
			gridv[0].append(y)
			gridv[1].append(int(y-(i+1)/2))
			gridv[1].append(y)
		elif y >= b-1 and i > 1:
			gridv[0].append(b)
			gridv[1].append(int(b-(i+1)/2))
			gridv[1].append(b)
			
			
	
	bigshiftx = random.choice([1,2,0]) #1
	bigseedx = np.ceil(len(gridh[0])/2.0)-bigshiftx-1
	bigshifty = random.choice([-3,2]) #-3
	bigseedy = np.ceil(len(gridh[0])/2.0) + bigshifty -1
	big = []
	
	#there are three large blocks: two 2x2 blocks and one 1x2 block
	#append coordinates of the corners of the large blocks to big
	big.append([bigseedx,bigseedy,bigseedx+1,bigseedy+1])
	big.append([bigseedx+3-5.0*(bigshiftx<2),bigseedy,bigseedx+3-5.0*(bigshiftx<2),bigseedy+1])
	big.append([bigseedx,bigseedy+4.0*(bigshifty<0)-4.0*(bigshifty>1),bigseedx+1,bigseedy+5.0*(bigshifty<0)-3.0*(bigshifty>1)])
	
	'''
	big.append([bigseedx,bigseedy,bigseedx+1,bigseedy+1])
	big.append([bigseedx+3,bigseedy,bigseedx+3,bigseedy+1])
	big.append([bigseedx,bigseedy+4,bigseedx+1,bigseedy+5])
	'''
	
	return gridh, gridv, big
	
	
	
	
	
'''
the colour selected for a block depends on the distance from the centre of the canvas and on the colour of neighbouring
blocks in contact with the current block
plotcolour = -1 denotes that the large block should not be filled; the original colour of the constituent blocks is retained
'''
def setplotcolour(gridh, gridv, l, i, j, extrax, extray, colour, occupied, neighbours, rad)	:
	if rad <= 0.25:
		plotcolour = random.choice(1*[-1]+ 20*list(set([0])-set(neighbours)) + 6*list(set([3]+range(5,len(colour)-1))-set(neighbours))) #random.randint(0, stop)
	elif rad <= 0.38:
		plotcolour = random.choice(1*list(set([-1])-set(neighbours))+6*list(set([0,1,2,3,4])-set(neighbours))+8*list(set(range(5,len(colour)-1))-set(neighbours))) #random.randint(0, stop)	
	elif rad <= 1.0:
		plotcolour = random.choice(12*list(set([1])-set(neighbours))+16*list(set([2,3])-set(neighbours))+4*list(set([4])-set(neighbours))+1*list(set(range(5,len(colour)-1))-set(neighbours))) #random.randint(0, stop)	
	else:
		plotcolour = len(colour)-1

	if plotcolour != -1:
		cplot[gridh[l][i]+extrax:gridh[l][i+1],gridv[l][j]+extray:gridv[l][j+1]] = plotcolour
		occupied[gridh[l][i]+extrax:gridh[l][i+1], gridv[l][j]+extray:gridv[l][j+1]] = l+1
				
	return cplot, occupied		
	
	
	
	
	
def adddot(cplot, occupied, gridh, gridv, ii, jj, l, colour, extrax, extray):
							
	minsize = 3
	if (l== 0 and ii < len(gridh[0])-2 and jj < len(gridv[0])-2) or l==1: # 
		if  gridh[l][ii+1]- gridh[l][ii] - extrax >=minsize and gridv[l][jj+1]- gridv[l][jj] -extray >=minsize and cplot[gridh[l][ii]+2,gridv[l][jj]+2] != 0:
			cplot[gridh[l][ii+1]-2,gridv[l][jj+1]-2] = random.choice([1,8,8]+range(5,len(colour)-1)) 
			occupied[gridh[l][ii+1]-2,gridv[l][jj+1]-2] = 4	
			#cplot[gridh[l][ii]+1+extrax,gridv[l][jj]+1+extray] = random.choice([1]+range(5,len(colour)-1)) 
			#occupied[gridh[l][ii]+1+extrax,gridv[l][jj]+1+extray] = 4	
		
	return cplot, occupied	
				




def fillgrid(gridh, gridv, posx, posy, colour, cplot, cutoffx, cutoffy, cutoffdot, addblocksoutsidegrid, gridshuffle):
	'''
	the "streets" belong to the left column and upper row of a block
	extrax, extray = 0: the "street" is covered
	extrax, extray = 1: there is a "street" between a block and its neighbour
	
	occupied = 0 for small blocks
	occupied = 1 for blocks on grid[1]
	occupied = 2 for blocks on grid[0]
	occupied = 3 for blocks in big
	occupied = 4 for dots on blocks
	
	blocks in contact cannot have the same colour
	neighbours contains the colours of blocks left or above the current block (where there is no "street" between the blocks)
	'''

	bw = len(posx)
	b = len(posy)
	
	occupied = np.zeros((bw,b), dtype=int)
	xextragrid = np.zeros((bw,b), dtype=int)
	yextragrid = np.zeros((bw,b), dtype=int)
	
	#gridh[0] columns which are further subdivided to make small blocks
	makesmall = [2,3, len(gridh[0])-5,len(gridh[0])-4]
	
	for i in range(len(gridh[0])-1):
		for j in range(len(gridv[0])-1):
		
			#distance of the centre of the block to the centre of the canvas, divided by the width of the canvas
			rad = np.abs((posx[gridh[0][i]]+1.0*posx[gridh[0][i+1]])/(2.0*width)-0.5)+ np.abs((posy[gridv[0][j]]+1.0*posy[gridv[0][j+1]])/(2.0*width)-0.5)
	
			for ii in [0,1]:
				for jj in [0,1]:
				
					'''
					plot the large blocks along the edges of the canvas and the large blocks denoted in big
					'''
					if ([i,j,i+ii, j+jj] in big or (i in [0,len(gridh[0])-3] and ii==1 and jj==1 and np.mod(j+1,2)) or (j in [0, len(gridv[0])-3]) and ii==1 and jj==1 and (np.mod(i+1,2) or i+2 ==len(gridh[0])-1 )) and j <=len(gridv[0])-3 and i <=len(gridh[0])-3: # :		
					
						if [i,j,i+ii, j+jj] in big:
							neighbours = get_neighbours(cplot,gridh, gridv, 0, i,j,0,1, occupied) + get_neighbours(cplot,gridh, gridv, 0, i,j+jj,0,1, occupied)
							if 0 in neighbours or 1 in neighbours:
								extrax = 1
							else:
								extrax = 0
							extray = 1
							plotcolour = 0 #force big central blocks to be white
						else:
							extrax = 0
							extray = 0	
							plotcolour = random.choice([2,3]) # force big peripheral blocks to be ivory or light grey
						
						# ensure that blocks in the first row extend to overlap the "street" of the row below
						if j==0 and jj==1:
							add = 1
						else:
							add = 0
						
						cplot[gridh[0][i]+extrax:gridh[0][i+ii+1],gridv[0][j]+extray:gridv[0][j+jj+1]+add] = plotcolour
						
						# note that the element at i,j is occupied by a very large block (3)
						occupied[gridh[0][i]+extrax:gridh[0][i+ii+1], gridv[0][j]+extray:gridv[0][j+jj+1]+add] = 3
	
							
	
					'''
					add the small blocks on top of the large blocks along the edges of the canvas
					'''
					if addblocksoutsidegrid:
						extrax = 0
						extray = 0
						if i+ii == 1:
							neighbours = get_neighbours(cplot,gridh, gridv, 1, 2*i,2*j,extrax, extray, occupied)
							cplot, occupied = setplotcolour(gridh, gridv, 1, 3, 2*j, extrax, extray, colour, occupied, neighbours, rad)
		
						if i == len(gridh[0])-3 and posx[gridh[1][3]] < width - posx[gridh[1][len(gridh[1])-4]]:
							neighbours =get_neighbours(cplot,gridh, gridv, 1, 2*i,2*j,extrax, extray, occupied)
							cplot, occupied = setplotcolour(gridh, gridv, 1, 2*len(gridh[0])-6, 2*j, extrax, extray, colour, occupied, neighbours, rad)
						
						if j == len(gridv[0])-3 and i < len(gridh[0])-2 and i >1:
							neighbours = get_neighbours(cplot,gridh, gridv, 1, 2*i,2*j,extrax, extray, occupied)+get_neighbours(cplot,gridh, gridv, 1, 2*i-1,2*j,extrax, extray, occupied)
							
							l = 1
							cplot, occupied = setplotcolour(gridh, gridv, 1, 2*i-1, 2*j, extrax, extray, colour, occupied, neighbours, rad)
							cplot[gridh[l][2*i]+extrax:gridh[l][2*i+1],gridv[l][2*j]+extray:gridv[l][2*j+1]] = cplot[gridh[l][2*i-1],gridv[l][2*j]+extray]
							occupied[gridh[l][2*i]+extrax:gridh[l][2*i+1],gridv[l][2*j]+extray:gridv[l][2*j+1]] = 2
						
											
			'''
			if the blocks at i,j are not occupied by a large block then
			draw the medium-sized blocks denoted by the rows and columns in grid[0]
			'''
			if occupied[gridh[0][i]+1,gridv[0][j]+1] == 0:
				if  not (i in makesmall  ): 
					rad = np.abs((posx[gridh[0][i]]+1.0*posx[gridh[0][i+1]])/(2.0*width)-0.5)+ np.abs((posy[gridv[0][j]]+1.0*posy[gridv[0][j+1]])/(2.0*width)-0.5)
	
					#decide whether to cover the "street"
					if rad > 0.33:
						extrax = 0
					else:	
						extrax = random.random() > 0.5*cutoffx + rad
					
					if (j >1 and i<bw-2 and isblockinbig(j, big,3)):		
						extray = 0
					elif j >1 and i<bw-2 and isblockinbig(j, big,1):		
						extray = 2	
					else:
						rand = random.random()
						extray = 1 #1.0*( rand > cutoffy*(2.5*rad)**2.0) + 1.0*(rand > cutoffy + 2*(1-cutoffy)*np.abs(rad-0.25)**0.25) #*(gridv[j+2]-gridv[j] > 3)
						
					neighbours = get_neighbours(cplot, gridh, gridv, 0, i,j, extrax, extray, occupied)	
					
					cplot, occupied = setplotcolour(gridh, gridv, 0, i, j, extrax, extray, colour, occupied, neighbours, rad)
	
					#add a dot to the block?
					if (random.random() < cutoffdot):
						adddot(cplot, occupied, gridh, gridv, i, j, 0, colour, extrax, extray)
						
					#shift the left side of the block to the left?	
					if gridshuffle:	
						if i >= 1 and occupied[gridh[0][i]+1,gridv[0][j]+1] >= occupied[gridh[0][i-1]+1,gridv[0][j]+1] and occupied[gridh[0][i],gridv[0][j]+1] !=0:	
							xextragrid[gridh[0][i],gridv[0][j]:gridv[0][j+1]] = np.round(width/450.0*random.choice([0,2,4]))
				
				
				else:
					'''
					if the blocks at i,j are not occupied by a large or medium-sized block then
					draw the small-sized blocks denoted by the rows and columns in grid[1].
					the rows of gridh[0] which are to be filled with small-sized blocks are stored in makesmall
					'''
					for ii in [2*i+0, 2*i+1]:
						for jj in [2*j+0, 2*j+1]:
						
							if np.abs(jj-len(gridv[1])/2.0)/1.0/len(gridv[1]) <0.2:  #if near the top or bottom of the canvas 
								extrax = random.random() > 0.9*cutoffx #0.8#
								extray = random.random() > 7.0*cutoffy
								
							else:
								extrax = 0
								extray = 0
								
							neighbours = get_neighbours(cplot,gridh, gridv, 1, ii,jj, extrax, extray, occupied)	
							rad = np.abs((posx[gridh[1][ii]]+1.0*posx[gridh[1][ii+1]])/(2.0*width)-0.5)+ np.abs((posy[gridv[1][jj]]+1.0*posy[gridv[1][jj+1]])/(2.0*width)-0.5)
		
							cplot, occupied = setplotcolour(gridh, gridv, 1, ii, jj, extrax, extray, colour, occupied, neighbours, rad)
		
							#add a dot to the block?
							if (random.random() < 1.5*cutoffdot):
								cplot, occupied = adddot(cplot, occupied, gridh, gridv, ii,jj, 1, colour, extrax, extray)
	
							#shift the left side of the block to the left?
							#shift the upper side of the block upwards?
							if gridshuffle:
								if ii >= 1 and occupied[gridh[1][ii]+1,gridv[1][jj]+1] == occupied[gridh[1][ii-1]+1,gridv[1][jj]+1] and occupied[gridh[1][ii-1]+1,gridv[1][jj]+1] !=0:	
									xextragrid[gridh[1][ii],gridv[1][jj]:gridv[1][jj+1]] = 0 #np.round(width/450.0*random.choice([1,2]))							
								if jj >= 1 and occupied[gridh[1][ii]+1,gridv[1][jj]+1] == occupied[gridh[1][ii]+1,gridv[1][jj-1]+1] and occupied[gridh[1][ii]+1,gridv[1][jj-1]+1] !=0:	
									yextragrid[gridh[1][ii]:gridh[1][ii+1],gridv[1][jj]] = np.round(width/450.0*random.choice([0,2,3]))							
	
	
	return cplot, occupied, xextragrid, yextragrid

	
	
	
	
	
	
	
	
	
'''
mondify: main program
'''

parser = argparse.ArgumentParser(description='Victory Boogie Woogie')
parser.add_argument('seed', action='store', nargs='?', type=int, default=0, help='seed for random number generator')
parser.add_argument('width', action='store', nargs='?', type=int, default=450, help='canvas width and height')
parser.add_argument('b', action='store', nargs='?', type=int, default=68, help='number of rows in the grid')
parser.add_argument('palette', choices=['grey', 'victory', 'orange'], action='store', default='victory', help='palette = grey, victory or boogie')
parser.add_argument('-noblocksoutsidegrid', dest='blocksoutsidegrid', action='store_false')
parser.add_argument('-nogridshuffle', dest='gridshuffle', action='store_false')
parser.add_argument('-notriangles', dest='drawtriangles', action='store_false')

args = parser.parse_args()

random.seed(args.seed)

width, b = args.width, args.b
if b < 12:
	b = 12
bw = int(64.0*b/70.0) # number of columns in the grid



cutoffx = 0.8 # probability of a central medium-sized block being in contact with left neighbour
cutoffy = 0.1 # probability of a central medium-sized block being in contact with upper neighbour
cutoffdot = 0.3 #0.2 # probability that there is a dot on a medium-sized block greater than 3x3

scale = b/5.0 #13.6#13.5 #dimension of medium-sized blocks (grid rows)

# palette
# white=0, ivory=1, lightgrey=2, grey=3,  goldenrod=4,yellow=5, red=6, blue=7, black=8 


if args.palette == 'grey':	
	colour = ["#FFFFFF",  "#f7faf7" ,"#DDDDDD", "#BBBBBB","#f7faf7" ,"#DDDDDD", "#BBBBBB","blue", "#000000"]
elif args.palette == 'orange':	
	colour = ["#FFFFFF",  "#f7faf7" ,"#DDDDDD", "#BBBBBB","#daa520", "#f8e800","#dd2222","orange", "#000000"]
else:
	colour = ["#FFFFFF",  "#f7faf7" ,"#DDDDDD", "#BBBBBB","#daa520", "#f8e800","#dd2222","blue", "#000000"]
	
# initialize canvas
master = Tk()
w = Canvas(master, width=width, height=width)
w.pack()

# generate and colour the small blocks
posx, posy = gen_grid(width, bw, b)
cplot = colour_grid(width,bw,b) 

# generate the medium grids (gridh[0][:], gridv[0][:]), and (gridh[1][:], gridv[1][:])
# the smaller blocks at the left and right of the image are aligned using the rows and columns in (gridh[1][:], gridv[1][:])
# the medium-sized blocks away from the sides of the image are aligned using the rows and columns in (gridh[0][:], gridv[0][:])
# the large-sized blocks in the middle of the image are aligned along the rows and columns in big
gridh, gridv, big = gen_coarsegrids(bw, b,scale)

# fill grid with the large blocks
cplot, occupied, xextragrid, yextragrid = fillgrid(gridh, gridv, posx, posy, colour, cplot, cutoffx, cutoffy, cutoffdot, args.blocksoutsidegrid, args.gridshuffle)

plot_canvas(posx, posy, cplot, occupied, colour, xextragrid, yextragrid, args.drawtriangles)	

filename = "mondify_"+str(args.seed)+"_b"+str(b)+"-"+args.palette
if not args.gridshuffle:
	filename = filename + "_nogridshuffle"
if not args.blocksoutsidegrid:
	filename = filename + "_noblocksoutsidegrid"	
if not args.drawtriangles:
	filename = filename + "_notriangles"		
filename = filename+".ps"

w.postscript(file=filename, height=width, width=width, colormode="color")
	
mainloop()				