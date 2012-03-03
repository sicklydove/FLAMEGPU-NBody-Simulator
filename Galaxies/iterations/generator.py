import random
import sys
sys.dont_write_bytecode=True
from particleDistribution import *
from math import *

class  ParticleAgent:
  def  __init__(self, particleId, xPos,  yPos,  zPos,  xVel,  yVel,  zVel,  mass,  isDark):
    self.particleId=particleId
    self.xPos=xPos
    self.yPos=yPos
    self.zPos=zPos
    self.xVel=xVel
    self.yVel=yVel
    self.zVel=zVel
    self.mass=mass
    self.isDark=isDark

  def  writeAgent(self):
    outStr=""
    outStr+='<xagent>'
    outStr+='<name>Particle</name>'
    outStr+='<id>'
    outStr+=str(self.particleId)
    outStr+='<mass>'
    outStr+=str(self.mass)
    outStr+='</mass>'
    outStr+='<isDark>0</isDark>'
    outStr+='<x>'
    outStr+=str(self.xPos)
    outStr+='</x>'
    outStr+='<y>'
    outStr+=str(self.yPos)
    outStr+='</y>'
    outStr+='<z>'
    outStr+=str(self.zPos)
    outStr+='</z>'
    outStr+='<xVel>'
    outStr+=str(self.xVel)
    outStr+='</xVel>'
    outStr+='<yVel>'
    outStr+=str(self.yVel)
    outStr+='</yVel>'
    outStr+='<zVel>'
    outStr+=str(self.zVel)
    outStr+='</zVel>'
    outStr+='</xagent>\r\n'
    return  outStr

class  Simulation:
  def  __init__(self,  filename):
    self.filename=filename

  def  initOutput(self):
    self.outputFile=open(self.filename,  'w')
    self.outputFile.write('<states>\r\n<itno>0</itno>\r\n')

  def  closeOutput(self):
    self.outputFile.write('</states>')
    self.outputFile.close()

  def  writeAgents(self,  ls):
    for  agent  in  ls:
      agentXML=agent.writeAgent()
      self.outputFile.write(agentXML)
      

class  ProbabilityDistribution:
  def  __init__(self,  distribType,  val1=None,  val2=None):
    if(distribType  is  ('gaussian'  or  'normal')):
      self.distribType='gaussian'

      if((val1 or val2)  is  None):
        print  "ERROR:  Must  define  a  mu  and  sigma  value  for  a  Gaussian  distribution"
        exit()

      else:
        self.mu=val1
        self.sigma=val2

    elif(distribType is ('fixed'  or  'set'  or  'static')):
      self.distribType='fixed'

      if(val1  is  None):
        print  "ERROR:  Must  define  a  set  value  for  a  fixed  distribution"
        exit()

      else:
        self.fixedVal=val1

    elif(distribType is 'circle'):
      self.distribType='circle'
      self.centre=val1
      self.radius=val2

    else:
      self.distribType='linear'

      if((val1  or  val2)  is  None):
        print  "ERROR:  Must  define  a  Min  and  Max  value  for  a  random  distribution"
        exit()

      else:
        self.minVal=val1
        self.maxVal=val2

  def  getItem(self):
    if(self.distribType  is  'gaussian'):
      return  random.gauss(self.mu,  self.sigma)

    elif(self.distribType  is  'linear'):
      return  random.uniform(self.minVal,  self.maxVal)

    elif(self.distribType  is  'fixed'):
      return  self.fixedVal

    elif(self.distribType is 'circle'):
      randRadius=random.uniform(0, self.radius)
      randxPos=random.uniform(self.centre[0]-randRadius, self.centre[0]+randRadius)

      xPosSq=(randxPos-self.centre[0])**2
      radiusSq=randRadius**2

      maxYMinusBSq=radiusSq-xPosSq

      maxY=quadratic(1, 0-(2*self.centre[1]), ((self.centre[1]**2)-maxYMinusBSq))[0]

      randyPos=random.uniform(self.centre[1], maxY)

      randyPosSq=(randyPos-self.centre[1])**2
     
      if(random.random()>0.5):
        randyPos=self.centre[1]+(self.centre[1]-randyPos)

      zMinusCSq=radiusSq-xPosSq-randyPosSq

      zMinusC=sqrt(zMinusCSq)

      if(random.random()>0.5):
        zPos=zMinusC+self.centre[2]
      else:
        zPos=self.centre[2]-zMinusC
      return (randxPos,randyPos,zPos)
    
  def getType(self):
    return self.distribType

#stackoverflow code
def quadratic(a, b, c):
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
      return []
    elif discriminant == 0:
      return [-b / (2*a)]
    else:
      root = sqrt(discriminant)
      ls = [(-b + root) / (2*a), (-b - root) / (2*a)]
      return filter(lambda x: x>0, ls)


if  __name__  ==  '__main__':

  simulation=Simulation('./0.xml')
  simulation.initOutput()

  sphere=ParticleDistribution(5000, True)
  sphereMasses=ProbabilityDistribution('fixed', 0.0)
  spherePositions=ProbabilityDistribution('circle', (1,1,1), 0.2)
  sphereVelocities=ProbabilityDistribution('fixed', 0.0)

  sphere.setMasses(sphereMasses)
  sphere.setPositions(spherePositions)
  sphere.setVelocities(sphereVelocities)

  simulation.writeAgents(sphere.getParticleAgents())
  simulation.closeOutput()

  #largeDistrib=ParticleDistribution(2000, True)
  #largeDistribMasses=ProbabilityDistribution('fixed', 0.02)
  #largeDistribxPositions=ProbabilityDistribution('linear', 2, 0)
  #largeDistribyPositions=ProbabilityDistribution('linear', 2, 0)
  #largeDistribzPositions=ProbabilityDistribution('fixed', 0)
  #largeDistribVelocities=ProbabilityDistribution('fixed', 0.0)
  #
  #largeDistrib.setMasses(largeDistribMasses)
  #largeDistrib.setPositions(largeDistribxPositions, largeDistribyPositions, largeDistribzPositions)
  #largeDistrib.setVelocities(largeDistribVelocities)

  #dwarf1=ParticleDistribution(300,True)
  #dwarf1Masses=ProbabilityDistribution('fixed', 0.03)
  #dwarf1Positions=ProbabilityDistribution('circle', (1,1,1), 0.03)
  #dwarf1Vels=ProbabilityDistribution('fixed',0)

  #dwarf1.setMasses(dwarf1Masses)
  #dwarf1.setPositions(dwarf1Positions)
  #dwarf1.setVelocities(dwarf1Vels)

  #dwarf2=ParticleDistribution(800,True)
  #dwarf2Masses=ProbabilityDistribution('fixed', 0.03)
  #dwarf2Positions=ProbabilityDistribution('circle', (0.3,1.2,1),0.07)
  #dwarf2Vels=ProbabilityDistribution('fixed',0)

  #dwarf2.setMasses(dwarf2Masses)
  #dwarf2.setPositions(dwarf2Positions)
  #dwarf2.setVelocities(dwarf2Vels)

  #dwarf3=ParticleDistribution(500,True)
  #dwarf3Masses=ProbabilityDistribution('fixed', 0.03)
  #dwarf3Positions=ProbabilityDistribution('circle', (1.75, 0.3,0.3), 0.05)
  #dwarf3Vels=ProbabilityDistribution('fixed',0)

  #dwarf3.setMasses(dwarf3Masses)
  #dwarf3.setPositions(dwarf3Positions)
  #dwarf3.setVelocities(dwarf3Vels)


  #distrib1Particles=largeDistrib.getParticleAgents()
  #dwarf1Particles=dwarf1.getParticleAgents()
  #dwarf2Particles=dwarf2.getParticleAgents()
  #dwarf3Particles=dwarf3.getParticleAgents()

  #simulation.writeAgents(distrib1Particles)
  #simulation.writeAgents(dwarf1Particles)
  #simulation.writeAgents(dwarf2Particles)
  #simulation.writeAgents(dwarf3Particles)
  #


  #simulation.closeOutput()
