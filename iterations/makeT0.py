import sys
sys.dont_write_bytecode=True
import particleAgent
from particleAgent import Simulation, ProbabilityDistribution, ParticleAgent


class ParticleDistribution:
 def __init__(self, numAgents, usingZAxis):
  self.numAgents=numAgents
  self.usingZAxis=usingZAxis
  self.particles={}
  self.massesSet=False
  self.positionsSet=False
  self.velocitiesSet=False

  for counter in range (0, self.numAgents):
   self.particles[counter]=[0,False,(0,0,0),(0,0,0)]

 def setMasses(self, massDistribution):
  for count in range (0, self.numAgents):
   self.particles[count][0]=massDistribution.getItem()

   #darkMatter
   self.particles[count][1]=False
   self.massesSet=True
 
 def setPositions(self, xDistrib, yDistrib=None, zDistrib=None):
  #If not set, use same distribution for all dimensions
  if((yDistrib and zDistrib) is None):
   yDistrib=xDistrib
   zDistrib=xDistrib

   if(not self.usingZAxis):
     zDistrib=('fixed', 0)

  if(yDistrib.getType() is 'circle'):
	 for i in range(0, self.numAgents):
	  coords=(xDistrib.getItem())
	  if(not self.usingZAxis):
	    coords=(coords[0],coords[1],0)

	  self.particles[i][2]=coords

  else:
    for i in range (0, self.numAgents):
      self.particles[i][2]=(xDistrib.getItem(), yDistrib.getItem(), zDistrib.getItem())

  self.positionsSet=True
 
 def setVelocities(self, xVelDistrib, yVelDistrib=None, zVelDistrib=None):

  #If not set, use same distribution for all dimensions
  if((yVelDistrib and zVelDistrib) is None):
   yVelDistrib=xVelDistrib
   zVelDistrib=xVelDistrib

  for i in range (0, self.numAgents):
   self.particles[i][3]=(xVelDistrib.getItem(), yVelDistrib.getItem(), zVelDistrib.getItem())

  self.velocitiesSet=True

 def makeAgentsFromFile(self, fileloc):
   agentsDict={}

   #File loading code
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

 def getParticleAgents(self):

  particleAgents=[]
  if(not(self.massesSet and self.positionsSet and self.velocitiesSet)):
   print "ERROR: Can't write particle positions until masses, positions and velocities have been set"
 
  else:
    for key, val in self.particles.iteritems():
      mass=val[0]
      isDark=val[1]
      xPos=val[2][0]
      yPos=val[2][1]
      zPos=val[2][2]
      xVel=val[3][0]
      yVel=val[3][1]
      zVel=val[3][2]
      thisParticle=ParticleAgent(key, xPos, yPos, zPos, xVel, yVel, zVel, mass, isDark)
      particleAgents.append(thisParticle)

    return particleAgents



if  __name__  ==  '__main__':

  simulation=Simulation('./0.xml')
  simulation.initOutput()

  largeDistrib=ParticleDistribution(5000, True)
  largeDistribMasses=ProbabilityDistribution('fixed', 0.02)
  largeDistribxPositions=ProbabilityDistribution('linear', 5, 0)

  largeDistribXVelocities=ProbabilityDistribution('linear', 1, -1)
  
  largeDistrib.setMasses(largeDistribMasses)
  largeDistrib.setPositions(largeDistribxPositions)
  largeDistrib.setVelocities(largeDistribXVelocities)


  dwarf1=ParticleDistribution(1300,True)
  dwarf1Masses=ProbabilityDistribution('fixed', 0.05)
  dwarf1Positions=ProbabilityDistribution('circle', (0.2,0.5,2), 0.1)
  dwarf1xVels=ProbabilityDistribution('linear',0,0.0)
  dwarf1yVels=ProbabilityDistribution('linear',0,0.0)
  dwarf1zVels=ProbabilityDistribution('fixed',0)

  dwarf1.setMasses(dwarf1Masses)
  dwarf1.setPositions(dwarf1Positions)
  dwarf1.setVelocities(dwarf1xVels, dwarf1yVels, dwarf1zVels)

  dwarf2=ParticleDistribution(700,True)
  dwarf2Masses=ProbabilityDistribution('fixed', 0.05)
  dwarf2Positions=ProbabilityDistribution('circle', (1.3,2.2,0.0),0.06)
  dwarf2xVels=ProbabilityDistribution('linear', 0, 0.0)
  dwarf2yVels=ProbabilityDistribution('linear',0,0.0)
  dwarf2zVels=ProbabilityDistribution('fixed',0)

  dwarf2.setMasses(dwarf2Masses)
  dwarf2.setPositions(dwarf2Positions)
  dwarf2.setVelocities(dwarf2xVels, dwarf2yVels, dwarf2zVels)

  dwarf3=ParticleDistribution(700,True)
  dwarf3Masses=ProbabilityDistribution('fixed', 0.05)
  dwarf3Positions=ProbabilityDistribution('circle', (3.75, 0.34, 1.3), 0.05)
  dwarf3Vels=ProbabilityDistribution('fixed',0)

  dwarf3.setMasses(dwarf3Masses)
  dwarf3.setPositions(dwarf3Positions)
  dwarf3.setVelocities(dwarf3Vels)

  dwarf4=ParticleDistribution(700,True)
  dwarf4Masses=ProbabilityDistribution('fixed', 0.05)
  dwarf4Positions=ProbabilityDistribution('circle', (1.75, 2.34, 3.3), 0.05)
  dwarf4Vels=ProbabilityDistribution('fixed',0)

  dwarf4.setMasses(dwarf3Masses)
  dwarf4.setPositions(dwarf4Positions)
  dwarf4.setVelocities(dwarf4Vels)


  dwarf5=ParticleDistribution(700,True)
  dwarf5Masses=ProbabilityDistribution('fixed', 0.05)
  dwarf5Positions=ProbabilityDistribution('circle', (3.75, 3.54, 3.1), 0.05)
  dwarf5Vels=ProbabilityDistribution('fixed',0)

  dwarf5.setMasses(dwarf5Masses)
  dwarf5.setPositions(dwarf5Positions)
  dwarf5.setVelocities(dwarf5Vels)


  dwarf6=ParticleDistribution(700,True)
  dwarf6Masses=ProbabilityDistribution('fixed', 0.05)
  dwarf6Positions=ProbabilityDistribution('circle', (4.75, 1.84, 4.3), 0.05)
  dwarf6Vels=ProbabilityDistribution('fixed',0)

  dwarf6.setMasses(dwarf6Masses)
  dwarf6.setPositions(dwarf6Positions)
  dwarf6.setVelocities(dwarf6Vels)


  distrib1Particles=largeDistrib.getParticleAgents()
  dwarf1Particles=dwarf1.getParticleAgents()
  dwarf2Particles=dwarf2.getParticleAgents()
  dwarf3Particles=dwarf3.getParticleAgents()
  dwarf4Particles=dwarf4.getParticleAgents()
  dwarf5Particles=dwarf5.getParticleAgents()
  dwarf6Particles=dwarf6.getParticleAgents()

  simulation.writeAgents(distrib1Particles)
  simulation.writeAgents(dwarf1Particles)
  simulation.writeAgents(dwarf2Particles)
  simulation.writeAgents(dwarf3Particles)
  simulation.writeAgents(dwarf4Particles)
  simulation.writeAgents(dwarf5Particles)
  simulation.writeAgents(dwarf6Particles)

  simulation.closeOutput()
