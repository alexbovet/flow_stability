#ifndef SPA_H
#define SPA_H

#include <vector> 

using namespace std; 

class SPA 
{ 
    public: 
    int size;
    std::vector<double> w ; // values
    std::vector<int> b; // switch, if == current row, position is occupied
    int current_row;
    std::vector<int> LS; // list of occupied col index
    
    SPA();
    SPA(int size) ;
    ~SPA ();
    void scatter(double value, int pos);
    void reset(int current_row);
    
}; 

#endif
