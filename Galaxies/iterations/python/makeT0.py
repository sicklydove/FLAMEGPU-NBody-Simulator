import sys
sys.dont_write_bytecode=True
from simulation import Simulation
from probabilityDistribution import ProbabilityDistribution
from particleDistribution import ParticleDistribution
from particleAgent import ParticleAgent

if  __name__  ==  '__main__':

  sim=Simulation('./0.xml')
  sim.initOutput()

  distrib=ParticleDistribution(8000, True, 100)
  distribMass=ProbabilityDistribution('fixed', 0.001)
  distribPos=ProbabilityDistribution('linear', -1.5, 1.5)
  distribVels=ProbabilityDistribution('linear', -0.5,0.5)
  distrib.setMasses(distribMass)
  distrib.setPositions(distribPos)
  distrib.setVelocities(distribVels) 
  sim.writeAgents(distrib.getParticleAgents())

  distrib=ParticleDistribution(500, True)
  distribMass=ProbabilityDistribution('fixed', 0.05)
  distribPos=ProbabilityDistribution('circle', (-0.5,-0.3,-0), 0.1)
  distribVels=ProbabilityDistribution('linear', -0.5,0.5)
  distrib.setMasses(distribMass)
  distrib.setPositions(distribPos)
  distrib.setVelocities(distribVels) 
  sim.writeAgents(distrib.getParticleAgents())

  distrib=ParticleDistribution(500, True)
  distribMass=ProbabilityDistribution('fixed', 0.05)
  distribPos=ProbabilityDistribution('circle', (1,0,1), 0.1)
  distribVels=ProbabilityDistribution('linear', -0.5,0.5)
  distrib.setMasses(distribMass)
  distrib.setPositions(distribPos)
  distrib.setVelocities(distribVels) 
  sim.writeAgents(distrib.getParticleAgents())

  sim.closeOutput()
