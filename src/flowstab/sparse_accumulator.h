/*
#
# flow stability
#
# Copyright (C) 2021 Alexandre Bovet <alexandre.bovet@maths.ox.ac.uk>
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef SPARSE_ACCUMULATOR_H
#define SPARSE_ACCUMULATOR_H

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

#endif // SPARSE_ACCUMULATOR_H
