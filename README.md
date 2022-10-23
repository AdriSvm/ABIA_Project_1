# Central distribution project



<h4 align="center">Program for the resolution of practice 1 of ABIA of the GIA degree of the UPC</h4>


<p align="center">
  <a href="#What-I'm-seeing-and-who-made-this">What I'm Seeing</a> •
  <a href="#how-to-download-and-run">How To Download</a> •
  <a href="#How-does-the-program-work">How Does It Work</a> •
  <a href="#license">License</a>
</p>


## What I'm seeing and who made this

* This is a development program for the resolution of practice 1 of the subject of Basic Algorithms of Artificial 
Intelligence of the third semester of the degree of artificial intelligence of the FIB UPC
* It has been created by a group of three Students:
    - Javier Puerta del valle javier.puerta@estudiantat.upc.edu
    - Daniel Spiridonov Poch daniel.spiridonov@estudiantat.upc.edu
    - Adrià Moya Morera adria.moya.morera@estudiantat.upc.edu
* Who are:
    - We are three students on the second year of the Artificial Intelligence degree, with enthusiasm for coding, computer science, technology and everything that involves AI.
* Where you can find us:
    - For any contact related of the useof the program, you can contact us by email
    - You can also find us in Linkedin:
        - Adrià Moya: www.linkedin.com/in/adrià-moya
        - Daniel Poch: www.linkedin.com/in/daniel-spiridonov-poch-a1b3021a7

## How to download and run 

To clone and run this application, you'll need [Git](https://git-scm.com) and [python3](https://www.python.org/) installed on your computer. From your command line:

You may need additional python libraries.
```bash
# Clone this repository
$ git clone https://github.com/AdriSvm/ABIA_Project_1.git

# Go into the repository
$ cd ABIA_Project_1

#Dive into files and explore
#For the graphical interface for Windows10-11
$ ./graphic

#If you want a more console based
$ python3
> from main import *

#In main file you can change experiment() for your personal use
```
## How does the program work
You have all the information about the problem in enunciat.pdf on the GitHub repository.

The main goal is to distribute n clients in k power plants the most efficient way for maximizing the benefits.
We have different types of clients and centrals as you'll see in enunciat.pdf, it's all based on an imaginary scenario.

We are using Hill Climbing and Simulated Annealing algorithms to solve the problem.

* The main parameters you'll have to set for solving the problem for a particular situation are:
    - Number of clients n: int(), Default: 1000
    - Clients proportions[XG,MG,G]: list[float,float,float], Deafault: [0.2,0.3,0.5]
    - Guaranteed proportions: float(range(0-1)), Default: 0.75
    - Power plants proportions[A,B,C]: list[int,int,int], Default: [5,10,25]
    - Seed: int(), Default: 1234
    - Solving algorithm["HILL CLIMBING", "SIMULATED ANNEALING"]: Default: "HILL CLIMBING"
    - Function for initial state["ONLY GRANTED","ORDERED"]: Defaul: "ONLY GRANTED"
  
* Also, there are two optional parameters:
    - Time[True,False]: Default: False
    - n_iter: int(), Default: 5

Time set to True if you want to see how much time does the program spend, it will output the total time, for seeing time for one execution, you'll need to do time / n_iter

Note: In the graphical version, depending on the parameters, program may spend a large quantity of time.

See informe.pdf for more information.

## Emailware

Central Distribution is an [emailware](https://en.wiktionary.org/wiki/emailware). Meaning, if you liked using this app or it has helped you in any way, I'd like you send me an email at any of our emails, about anything you'd want to say about this software. I'd really appreciate it!

## License

MIT

---

> Adrià Moya &nbsp;&middot;&nbsp;
> GitHub [@AdriSvm](https://github.com/AdriSvm) &nbsp;&middot;&nbsp;
> Twitter [@Adriasvm2](https://twitter.com/adriasvm2)

> Javier Puerta &nbsp;&middot;&nbsp;
> GitHub [@Javierpuerta22](https://github.com/Javierpuerta22) &nbsp;&middot;&nbsp;
> Twitter [@Javi_p22](https://twitter.com/Javi_p22)

> Daniel S.Poch &nbsp;&middot;&nbsp;
> GitHub [@danielpoch](https://github.com/danielpoch) &nbsp;&middot;&nbsp;
> Twitter [@DanielSPoch](https://twitter.com/DanielSPoch)

