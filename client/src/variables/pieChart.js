import React from 'react';
import {Pie} from 'react-chartjs-2';
import {
    Card,
    CardHeader,
    CardBody,
    Container,
    Row,
    Col,
  } from "reactstrap";

const state1 = {
  labels: ['YES', 'NO'],
  datasets: [
    {
      label: 'Online order',
      backgroundColor: [
        '#B21F00',
        '#2FDE00',
      ],
      hoverBackgroundColor: [
      '#501800',
      '#175000',
      ],
      data: [58.87, 41.13]
    }
  ]
}



const state2 = {
  labels: ['YES', 'NO'],
  datasets: [
    {
      label: 'online table bookings',
      backgroundColor: [
        '#00A6B4',
        '#C9DE00',
      ],
      hoverBackgroundColor: [
      '#003350',
      '#4B5000',
      ],
      data: [12.47, 87.53]
    }
  ]
}


const PieChart = () => {

    return (
      <div>
        <Container>
        <Row>
          <Col className="mb-5 mb-xl-0" xl="6">
            <Card className="bg-gradient-default shadow">
              <CardHeader className="bg-transparent">
                <Row className="align-items-center">
                  <div className="col">
                    <h6 className="text-uppercase text-light ls-1 mb-1">
                      Overview
                    </h6>
                    <h2 className="text-white mb-0">Accepting Vs Not Accepting</h2>
                  </div>
                </Row>
              </CardHeader>
              <CardBody>
              <Pie
                data={state1}
                options={{
                    maintainAspectRatio: true,
                    title:{
                    display:true,
                    text:'Online order',
                    fontSize:20
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
          </Col>
          <Col xl="6">
          <Card className="bg-gradient-default shadow">
              <CardHeader className="bg-transparent">
                <Row className="align-items-center">
                  <div className="col">
                    <h6 className="text-uppercase text-light ls-1 mb-1">
                      Overview
                    </h6>
                    <h2 className="text-white mb-0">Accepting Vs Not Accepting</h2>
                  </div>
                </Row>
              </CardHeader>
              <CardBody>
              <Pie
                data={state2}
                options={{
                    maintainAspectRatio: true,
                    title:{
                    display:true,
                    text:'Online booking table',
                    fontSize:20
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
          </Col>
        </Row>
        </Container>
      </div>
    );
}

export default PieChart;