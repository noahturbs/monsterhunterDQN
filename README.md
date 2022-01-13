This program runs a machine learning A.I. that plays monster hunter tri on the dolphin emulator.


To run this you'll need:

-monster hunter tri iso file

-dolphin emulator

-AHK(AutoHotKey) installed via their installer. use https://www.autohotkey.com/download/



Main takeaways from doing this project:
machine learning was surprisingly the most straight forward thing to learn, both technically and abstract-wise.

but everything else about this project was surprisingly difficult: virtual keyboards and reverse engineering memory.


-virtual keyboards lack some of the 'permissions' of real keyboards. Basically another way for software to identify between bots and real people.
took awhile to diagnose the difference, and some more time in finding a solution (AHK can emulate these hardware hooks).
	
-reverse engineering memory was rather straightforward to learn... but sifting through relevant values to get access was annoyingly time consuming.
	
-Plan things out before writing code; makes readability way better. I finished a software design course in between starting and finishing this project.
I took for granted reading other people's code, APIs with good documentation, and my ability to read my own code. Good coding practices start before writing any code.


main article used for ML: https://towardsdatascience.com/creating-ai-for-gameboy-part-4-q-learning-and-variations-2a0d35dd0b1c
