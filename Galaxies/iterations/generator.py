import random


class ParticleAgent:
  def __init__(self, particleId, xPos, yPos, zPos, xVel, yVel, zVel, mass, isDark):
    self.particleId=particleId
    self.xPos=xPos
    self.yPos=yPos
    self.zPos=zPos
    self.xVel=xVel
    self.yVel=yVel
    self.zVel=zVel
    self.mass=mass
    self.isDark=isDark

  def writeAgent(self):
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
    return outStr

class Simulation:
  def __init__(self, outputFile, numAgents, maxMass, maxVel, minVel, maxPos, minPos, randomMass, randomVel, usingZAxis):
    self.numAgents=numAgents
    self.maxMass=maxMass
    self.maxVel=maxVel
    self.minVel=minVel
    self.maxPos=maxPos
    self.minPos=minPos
    self.randomMass=randomMass
    self.randomVel=randomVel
    self.usingZAxis=usingZAxis
    self.outputFile=outputFile

  def initOutput(self):
    self.outputFile.write('<states>\r\n<itno>0</itno>\r\n')

  def closeOutput(self):
    self.outputFile.write('</states>')

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

  def makeAgents(self):
    for num in range(0,self.numAgents):
      isDark=False

      positionsAndVelocities=[0,0,0,0,0,0]

      for index in range(0,3):
	val=random.uniform(self.minPos, self.maxPos)
        positionsAndVelocities[index]=val

      for index in range(3,6):
	val=random.random()
	boolean=random.random()
	if(boolean>0.5):
		val=0-val
        positionsAndVelocities[index]=val*self.maxVel
        if(not (self.randomVel)):
          positionsAndVelocities[index]=maxMass

      if(not(self.usingZAxis)):
        positionsAndVelocities[2]=0
        positionsAndVelocities[5]=0

      mass=maxMass
      if(self.randomMass):
	mass=random.random()
	mass=self.maxMass*mass

      xPos=positionsAndVelocities[0]
      yPos=positionsAndVelocities[1]
      zPos=positionsAndVelocities[2]
      xVel=positionsAndVelocities[3]
      yVel=positionsAndVelocities[4]
      zVel=positionsAndVelocities[5]

      newAgent=ParticleAgent(num, xPos, yPos, zPos, xVel, yVel, zVel, mass, isDark)
      agentXML=newAgent.writeAgent()
      self.outputFile.write(agentXML)

if __name__ == '__main__':
  numAgents=1000
  maxMass=0.05
  maxVel=0.7
  maxPos=0.3
  minPos=0
  randomMass=False
  randomVel=True
  usingZAxis=True
  outFile=open('./0.xml', 'w')
  sim=Simulation(outFile, numAgents, maxMass, maxVel, maxVel, maxPos, minPos, randomMass, randomVel, usingZAxis)
  distrib=Simulation(outFile, 5000, 0.05, 0.7, 0.7, .7, -1.25, False, randomVel, usingZAxis)
  sim.initOutput()

  #sim.makeAgentsFromFile('tab8096.txt')
  sim.makeAgents()
  distrib.makeAgents()

  sim.closeOutput()
  outFile.close()

