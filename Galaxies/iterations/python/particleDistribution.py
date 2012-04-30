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

   #Particles: [mass, (xyz pos), (xyz vel)]
   numParticlesPerGroup=numAgents/numParticleGroups
   particleGroup=1

   for counter in range (0, self.numAgents):
     particleGroup+=1
     if(particleGroup>numParticleGroups):
       particleGroup=1

     #Dark Matter. No booleans in FLAMEGPU, so use 0/1
     if (random.random() > prob):
       isDark=0 
     else:
       isDark=1
     self.particles.append(ParticleAgent(0,isDark,particleGroup,(0,0,0),(0,0,0)))

  def setMasses(self, massDistribution):

    #catch and stop people setting masses as spheres.
    #This won't error at runtime b/c values passed as tuples -> interpreted as null (0)
    if(massDistribution.getType() is 'sphere'):
      print "Error! Can't set masses with a spherical distribution"
      exit()

    for count in range (0, self.numAgents):
      self.particles[count].setMass(massDistribution.getItem())

  def setPositions(self, xDistrib, yDistrib=None, zDistrib=None):
    #If not set, use same distribution for all dimensions
    if((yDistrib and zDistrib) is None):
      yDistrib=xDistrib
      zDistrib=xDistrib

    if(not self.usingZAxis):
      zDistrib=ProbabilityDistribution('fixed', 0)

    if(xDistrib.getType() is 'sphere'):
      if(xDistrib.getRadius()<0):
        print "Error! Can't define a spherical distribution with negative radius"
        exit()
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

    #catch and stop people setting velocities as spheres.
    #This won't error at runtime b/c values passed as tuples -> interpreted as null (0)
    if((xVelDistrib.getType() is 'sphere') or (yVelDistrib.getType() is 'sphere') or (zVelDistrib.getType() is 'sphere')):
      print "Error! Can't set velocities with a spherical distribution"
      exit()

    if(not self.usingZAxis):
      zVelDistrib=ProbabilityDistribution('fixed',0)

    for i in range (0, self.numAgents):
      self.particles[i].setVels(xVelDistrib.getItem(), yVelDistrib.getItem(), zVelDistrib.getItem())

  def getParticleAgents(self):
    return self.particles
