import random
f = open ('./0.xml', 'w')
string='<states>'
string+='<itno>0</itno>'
for x in range(1,20000):
	string+'<xagent>'
	string+='<name>Particle</name>'
	string+='<id>'
	string+=str(x)
	string+='<mass>'
	string+='1'
#	string+=str(10*random.random())
	string+='</mass>'
	string+='<isDark>0</isDark>'
	string+='<x>'
	#string+='0'
	string+=str(200*random.random())
	string+='</x>'
	string+='<y>'
	#string+='0'
	string+=str(200*random.random())
	string+='</y>'
	string+='<z>'
#	string+='0'
	string+=str(200*random.random())
	string+='</z>'
	string+='<xVel>'
	#string+='0'
	var=random.random()
#	if(var>0.5):
#		string+=str(150*var)
#	else:
#		string+=str(0-150*var)
	string+='0'
	string+='</xVel>'
	string+='<yVel>'
	string+='0'
	var=random.random()
#	if(var>0.5):
#		string+=str(150*var)
#	else:
#		string+=str(0-150*var)
	string+='</yVel>'
	string+='<zVel>'
	var=random.random()
	#if(var>0.5):
	#	string+=str(1*var)
	#else:
	#	string+=str(0-1*var)
	string+='0'
	string+='</zVel>'
	string+='</xagent>\r'
string+='</states>'
f.write(string);

