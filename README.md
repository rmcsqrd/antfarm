## AntFarm
Force-based motion planning algorithm with RL navigational feedback. Implementation based on [Force-based Algorithm for Motion Planning of Large Agent Teams](https://arxiv.org/pdf/1909.05415.pdf).

![CircleSwap](https://github.com/rmcsqrd/antfarm/blob/master/aux/readme/simresult.gif)

![movingline](https://github.com/rmcsqrd/antfarm/blob/master/aux/readme/simresult_movingline.gif)
### Usage
```
    $ cd path/to/your/directory
    $ julia
    $ julia> ]
    $ (v1.x) pkg> activate . 
    $ julia> using antfarm
    $ julia> antfarm.[command]()  // this is how it works
```
## Note on Branches
- Master: Old Version
- CSCI 5454: Algorithm version for CSCI 5454 Class
- RL-cpu: Thesis Version
