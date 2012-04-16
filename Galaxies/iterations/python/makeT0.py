import sys
sys.dont_write_bytecode=True
from simulation import Simulation
from probabilityDistribution import ProbabilityDistribution
from particleDistribution import ParticleDistribution
from particleAgent import ParticleAgent

if  __name__  ==  '__main__':

  sim=Simulation('./0.xml')
  sim.initOutput()

  distrib=ParticleDistribution(16000, True, 0, 1)
  distribMass=ProbabilityDistribution('fixed', 0.001)
  distribPos=ProbabilityDistribution('linear', -2, 2)
  distribxVels=ProbabilityDistribution('fixed', -0.0)
  distribyVels=ProbabilityDistribution('fixed', -0.0)
  distribzVels=ProbabilityDistribution('fixed', -0.0)

  distrib.setMasses(distribMass)
  distrib.setPositions(distribPos)
  distrib.setVelocities(distribxVels,distribyVels,distribzVels) 
  sim.writeAgents(distrib.getParticleAgents())

  sim.closeOutput()
