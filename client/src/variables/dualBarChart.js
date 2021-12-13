import React from 'react';
import {Bar, HorizontalBar} from 'react-chartjs-2';

import {
    Card,
    CardHeader,
    CardBody,
    Container,
    Row,
  } from "reactstrap";


/*
Quick Bites                   19132
Casual Dining                 10330
Cafe                           3732
Delivery                       2604
Dessert Parlor                 2263
Takeaway, Delivery             2037
Casual Dining, Bar             1154
Bakery                         1141
Beverage Shop                   867
Bar                             697
Food Court                      624
Sweet Shop                      468
Bar, Casual Dining              425
Lounge                          396
Pub                             357
Fine Dining                     346
Casual Dining, Cafe             319
Beverage Shop, Quick Bites      298
Bakery, Quick Bites             289
Mess                            267
*/

const state1 = {
  labels: [ 'North India',
            'North Indian, Chinese',
            'South Indian',
            'Biryani',
            'Bakery, Desserts',
            'Fast Food',
            'Desserts',
            'Cafe',
            'South Indian, North Indian, Chinese',
            'Bakery',
            'Chinese',
            'Ice Cream, Dessert',
            'Chinese, North Indian',
            'Mithai, Street Food',
            'Desserts, Ice Cream',
            'North Indian, Chinese, Biryani',
            'South Indian, North Indian',
            'North Indian, South Indian, Chinese',
            'Beverages'],
  datasets: [
    {
      label: 'Cuisine',
      backgroundColor: 'rgba(75,192,192,1)',
      borderColor: 'rgba(0,0,0,1)',
      borderWidth: 2,
      data: [2913, 2385, 1828, 918, 911, 803, 766, 756, 726, 651, 556, 417, 415, 372, 354, 352, 343, 305, 301]
    }
  ]
}

const state2 = {
  labels: [
    'Quick Bites',
    'Casual Dining',
    'Cafe',
    'Delivery',
    'Dessert Parlor',
    'Takeaway, Delivery',
    'Casual Dining, Bar',
    'Bakery',
    'Beverage Shop',
    'Bar',
    'Food Court',
    'Sweet Shop',
    'Bar, Casual Dining',
    'Lounge',
    'Pub',
    'Fine Dining',
    'Casual Dining, Cafe',
    'Beverage Shop, Quick Bites',
    'Bakery, Quick Bites',
    'Mess',
  ],
  datasets: [
    {
      label: 'Restaurant types',
      backgroundColor: 'rgba(75,192,192,1)',
      borderColor: 'rgba(0,0,0,1)',
      borderWidth: 2,
      data: [19132, 10330, 3732, 2604, 2263, 2037, 1154, 1141, 867, 697, 624, 468, 425, 396, 357, 346, 319, 298, 289, 267]
    }
  ]
}


const DualBarChart = () => {
    return (
      <div>
        <Container >
              <Card className="bg-gradient-default shadow">
              <CardHeader className="bg-transparent">
                  <Row className="align-items-center">
                  <div className="col">
                      <h6 className="text-uppercase text-light ls-1 mb-1">
                      Overview
                      </h6>
                      <h2 className="text-white mb-0">Popular Restaurant types in Bangalore by number</h2>
                  </div>
                  </Row>
              </CardHeader>
              <CardBody>
              <HorizontalBar
                  data={state2}
                  options={{
                      maintainAspectRatio: true,
                      title:{
                      display:true,
                      text:'Restaurant types',
                      fontSize:15
                      },
                      legend:{
                        labels: {
                          usePointStyle: false,
                          padding: 16,
                        },
                      display:true,
                      position:'right'
                      }
                  }}
              />
              </CardBody>
              </Card>
        </Container>
        
                
        <br />
        <br />

        <Container >
              <Card className="bg-gradient-default shadow">
              <CardHeader className="bg-transparent">
                  <Row className="align-items-center">
                  <div className="col">
                      <h6 className="text-uppercase text-light ls-1 mb-1">
                      Overview
                      </h6>
                      <h2 className="text-white mb-0">Popular Cuisines in Bangalore by number</h2>
                  </div>
                  </Row>
              </CardHeader>
              <CardBody>
              <Bar
                  data={state1}
                  options={{
                      maintainAspectRatio: true,
                      title:{
                      display:true,
                      text:'Cuisines',
                      fontSize:15,
                      },
                      legend:{
                        labels: {
                          usePointStyle: false,
                          padding: 16,
                        },
                      display:true,
                      position:'right'
                      }
                  }}
              />
              </CardBody>
              </Card>
        </Container>

      </div>
    );
}

export default DualBarChart;