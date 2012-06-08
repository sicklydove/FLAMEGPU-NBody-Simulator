class ParticleAgent:
  def __init__(self, mass, isDark, particleGroup, (xPos, yPos, zPos), (xVel, yVel, zVel)):
    self.xPos=xPos
    self.yPos=yPos
    self.zPos=zPos
    self.xVel=xVel
    self.yVel=yVel
    self.zVel=zVel
    self.mass=mass
    self.isDark=isDark
    self.particleGroup=particleGroup

  #Return XML for writing to t0 files
  def getAgentXML(self):
    outStr=""
    outStr+='<xagent>'
    outStr+='<name>Particle</name>'
    outStr+='<mass>'
    outStr+=str(self.mass)
    outStr+='</mass>'
    outStr+='<isDark>'
    outStr+=str(self.isDark)
    outStr+='</isDark>'
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
    outStr+='<particleGroup>'
    outStr+=str(self.particleGroup)
    outStr+='</particleGroup>'
    outStr+='</xagent>\r\n'
    return outStr

  def setMass(self, mass):
    self.mass=mass

  def setIsDark(self, isDark):
    self.isDark=isDark

  def setParticleGroup(self, offset):
    self.particleGroup=offset
  
  def setPositions(self, x,y,z):
    self.xPos=x
    self.yPos=y
    self.zPos=z

  def setVels(self, x,y,z):
    self.xVel=x
    self.yVel=y
    self.zVel=z

