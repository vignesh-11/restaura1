import React from 'react';
import {Bar} from 'react-chartjs-2';

import {
    Card,
    CardHeader,
    CardBody,
    Container,
    Row,
  } from "reactstrap";

/*
BTM                      5124
HSR                      2523
Koramangala 5th Block    2504
JP Nagar                 2235
Whitefield               2144
Indiranagar              2083
Jayanagar                1926
Marathahalli             1846
Bannerghatta Road        1630
Bellandur                1286
Electronic City          1258
Koramangala 1st Block    1238
Brigade Road             1218
Koramangala 7th Block    1181
Koramangala 6th Block    1156
Sarjapur Road            1065
Ulsoor                   1023
Koramangala 4th Block    1017
MG Road                   918
Banashankari              906
*/


const state = {
  labels: [
    'BTM',
    'HSR',
    'Koramangala 5th Block',
    'JP Nagar',
    'Whitefield',
    'Indiranagar',
    'Jayanagar',
    'Marathahalli',
    'Bannerghatta Road',
    'Bellandur',
    'Electronic City',
    'Koramangala 1st Block',
    'Brigade Road',
    'Koramangala 7th Block',
    'Koramangala 6th Block',
    'Sarjapur Road',
    'Ulsoor',
    'Koramangala 4th Block',
    'MG Road',
    'Banashankari',
  ],
  datasets: [
    {
      label: 'Location',
      backgroundColor: 'rgba(75,192,192,1)',
      borderColor: 'rgba(0,0,0,1)',
      borderWidth: 2,
      data: [5124, 2523, 2504, 2235, 2144, 2083, 1926, 1846, 1630, 1286, 1258, 1238, 1218, 1181, 1156, 1065, 1023, 1017, 918, 906]
    }
  ]
}


const BarChart = () => {
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
                      <h2 className="text-white mb-0">Foodie Areas in Bangalore</h2>
                  </div>
                  </Row>
              </CardHeader>
              <CardBody>
              <Bar
                  data={state}
                  options={{
                      maintainAspectRatio: true,
                      title:{
                      display:true,
                      text:'Location vs Number',
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

export default BarChart
