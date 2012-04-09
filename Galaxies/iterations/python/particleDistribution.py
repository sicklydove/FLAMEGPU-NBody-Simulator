import sys
sys.dont_write_bytecode=True
from probabilityDistribution import ProbabilityDistribution
from particleAgent import *
import random

class ParticleDistribution:
  def __init__(self, numAgents, usingZAxis=True, darkMatterPercentage=0, numParticleGroups=1):
   self.numAgents=numAgents
   self.usingZAxis=usingZAxis
   self.particles=[]
   self.darkMatterPercentage=darkMatterPercentage
   self.numParticleGroups=numParticleGroups
   prob=self.darkMatterPercentage/100

   #Particles: [id, mass, (xyz pos), (xyz vel)]
   for counter in range (0, self.numAgents):
     #Dark Matter. No booleans in FLAMEGPU, so use 0/1
     if (random.random() > prob):
       isDark=0 
     else:
       isDark=1
     particleGroup=int(random.uniform(0,numParticleGroups-1))

     self.particles.append(ParticleAgent(counter,0,isDark,particleGroup,(0,0,0),(0,0,0)))

  def setMasses(self, massDistribution):
    for count in range (0, self.numAgents):
      self.particles[count].setMass(massDistribution.getItem())

  def setPositions(self, xDistrib, yDistrib=None, zDistrib=None):
    #If not set, use same distribution for all dimensions
    if((yDistrib and zDistrib) is None):
      yDistrib=xDistrib
      zDistrib=xDistrib
    if(not self.usingZAxis):
      zDistrib=ProbabilityDistribution('fixed', 0)

    if(yDistrib.getType() is 'circle'):

      #if(not(yDistrib and zDistrib) is None):
       # print 'Error! Cannot specify multiple position distributions if using a circle'
        #exit()
      for i in range(0, self.numAgents):
        coords=(xDistrib.getItem())
        if(not self.usingZAxis):
          coords=(coords[0],coords[1],0)
        self.particles[i].setPositions(coords[0], coords[1], coords[2])
    else:	
      for i in range (0, self.numAgents):
        self.particles[i].setPositions(xDistrib.getItem(), yDistrib.getItem(), zDistrib.getItem())

  def setVelocities(self, xVelDistrib, yVelDistrib=None, zVelDistrib=None):
    #If not set, use same distribution for all dimensions
    if((yVelDistrib and zVelDistrib) is None):
      yVelDistrib=xVelDistrib
      zVelDistrib=xVelDistrib

    if(not self.usingZAxis):
      zVelDistrib=ProbabilityDistribution('fixed',0)

    for i in range (0, self.numAgents):
      self.particles[i].setVels(xVelDistrib.getItem(), yVelDistrib.getItem(), zVelDistrib.getItem())

  def getParticleAgents(self):
    return self.particles

  def makeAgentsFromFile(self, fileloc):
    agentsDict={}

    scaleFactor=1
    velScaleFactor=1
    massScale=1
    inFile=open(fileloc, 'r')

    count=0
    with inFile as f:
      content=f.readlines()

    for line in content:
      stripStr=line.split()
      massPosVel=[0,0,0,0,0,0,0]
      massPosVel[0]=massScale*float(stripStr[0])

    for index in range(1,4):
      massPosVel[index]=scaleFactor*float(stripStr[index])

    for index in range(4,7):
      massPosVel[index]=velScaleFactor*float(stripStr[index])

    count+=1

    agentsDict[count]=massPosVel

    for item in agentsDict:
      ls=agentsDict[item]
      newAgent=ParticleAgent(item, ls[1], ls[2], ls[3], ls[4], ls[5], ls[6], ls[0], False)
      agentXML=newAgent.writeAgent()
      self.outputFile.write(agentXML)
