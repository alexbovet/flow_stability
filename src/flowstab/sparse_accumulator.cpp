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

#include "sparse_accumulator.h"

using namespace std; 
    
SPA::SPA() {} 
//Parametrized Constructor 
SPA::SPA(int size) 
{
    this->size = size;
    
    this->w.resize(size);
    this->b.resize(size);
    
    for (int i = 0; i < size; i++) {
        this->w[i] = 0.0;
        this->b[i] = -1;
    }
    
    this->current_row = 0;
    
}
//Destructor
SPA::~SPA () {}

void SPA::scatter(double value, int pos){

    if (this->b[pos] < this->current_row) {
        this->w[pos] = value;
        this->b[pos] = this->current_row;
        this->LS.push_back(pos);
    }
    else {
        this->w[pos] += value;
    }
        
}

void SPA::reset(int current_row) {
    this->current_row = current_row;
    this->LS.clear();
}




