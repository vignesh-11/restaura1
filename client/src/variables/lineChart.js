import React from 'react';
import {Line} from 'react-chartjs-2';

import {
    Card,
    CardHeader,
    CardBody,
    Container,
    Row,
  } from "reactstrap";


const state = {
  labels: [40,
          50,
          70,
          80,
          100,
          120,
          130,
          150,
          160,
          180,
          199,
          200,
          230,
          240,
          250,
          300,
          330,
          350,
          360,
          400,
          450,
          500,
          550,
          560,
          600,
          650,
          700,
          750,
          800,
          850,
          900,
          950,
          1000,
          1050,
          1100,
          1200,
          1250,
          1300,
          1350,
          1400,
          1450,
          1500,
          1600,
          1650,
          1700,
          1800,
          1900,
          2000,
          2100,
          2200,
          2300,
          2400,
          2500,
          2600,
          2700,
          2800,
          3000,
          3200,
          3400,
          3500,
          3700,
          4000,
          4100,
          4500,
          5000,
          6000],
  datasets: [
    {
      label: 'Cost',
      backgroundColor: 'rgba(75,192,192,1)',
      borderColor: 'rgba(0,0,0,1)',
      borderWidth: 2,
      data: [
        8,
        6,
        3,
        4,
        702,
        2,
        8,
        1432,
        1,
        17,
        4,
        3527,
        10,
        2,
        2293,
        5735,
        4,
        1416,
        1,
        5562,
        1267,
        4326,
        719,
        1,
        3365,
        752,
        1873,
        749,
        2202,
        162,
        677,
        62,
        1566,
        4,
        510,
        979,
        9,
        515,
        18,
        473,
        5,
        947,
        266,
        6,
        247,
        203,
        70,
        356,
        67,
        78,
        11,
        23,
        146,
        10,
        3,
        45,
        162,
        2,
        13,
        25,
        1,
        29,
        4,
        2,
        1,
        2,
      ]
    }
  ]
}


const LineChart = () => {
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
                      <h2 className="text-white mb-0">Cost Distribution</h2>
                  </div>
                  </Row>
              </CardHeader>
              <CardBody>
              <Line
                  data={state}
                  options={{
                      maintainAspectRatio: true,
                      title:{
                      display:true,
                      text:'Cost Vs Number',
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
      </div>
    );
}

export default LineChart