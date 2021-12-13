import { Card, CardBody, CardTitle, Container, Row, Col } from "reactstrap";

const Header = ({rest}) => {
  return (
    <>
      <div className="header bg-gradient-info pb-8 pt-5 pt-md-8">
        <Container fluid>
          <div className="header-body">
            {/* Card stats */}
            <Row>
              <Col lg="6" xl="3">
                <Card className="card-stats mb-4 mb-xl-0">
                  <CardBody>
                    <Row>
                      <div className="col">
                        <CardTitle
                          tag="h5"
                          className="text-uppercase text-muted mb-0"
                        >
                          {rest["dining or delivery"]}
                        </CardTitle>
                        <span className="h2 font-weight-bold mb-0">
                          {rest["name"]}
                        </span>
                      </div>
                    </Row>
                    <p className="mt-3 mb-0 text-muted text-sm">
                      {rest["locality_address"]}
                    </p>
                  </CardBody>
                </Card>
              </Col>
              <Col lg="6" xl="3">
                <Card className="card-stats mb-4 mb-xl-0">
                  <CardBody>
                    <Row>
                      <div className="col">
                        <CardTitle
                          tag="h5"
                          className="text-uppercase text-muted mb-0"
                        >
                          Overall Rating dining: {rest["dining_rating"] === 0 ? "NA" : rest["dining_rating"]}
                          <br></br>
                          Overall Rating delivery: {rest["delivery_rating"] === 0 ? "NA" : rest["delivery_rating"]}
                        </CardTitle>
                        <br />
                        <span className="h2 font-weight-bold mb-0"><p>Cuisines: {rest["cuisine"]}</p></span>
                      </div>
                    </Row>
                    <p className="font-weight-bold text-muted text-md">
                      <span className="text-nowrap">Cost for 2: &#x20b9;{rest["cost"]}</span>
                    </p>
                  </CardBody>
                </Card>
              </Col>
              <Col lg="6" xl="3">
                <Card className="card-stats mb-4 mb-xl-0">
                  <CardBody>
                    <Row>
                      <div className="col">
                        <CardTitle
                          tag="h5"
                          className="text-uppercase text-muted mb-0"
                        >
                          Dining Performance 
                        </CardTitle>
                        <span className="h2 font-weight-bold mb-0">{
                       rest["dining or delivery"] === "Dining and Delivery" || rest["dining or delivery"] === "Dining"
                        ?
                          Object.values(rest["performance_dining"][rest["performance_dining"].length - 1])
                        :
                          "NA"
                        }</span>
                      </div>
                      <Col className="col-auto">
                        <div className="icon icon-shape bg-yellow text-white rounded-circle shadow">
                          <i className="fas fa-users" />
                        </div>
                      </Col>
                    </Row>
                    {
                       rest["dining or delivery"] === "Dining and Delivery" || rest["dining or delivery"] === "Dining"
                        ?
                        (<p className="mt-3 mb-0 text-muted text-sm">
                          {Object.values(rest["performance_dining"][rest["performance_dining"].length - 1]) 
                          -Object.values(rest["performance_dining"][rest["performance_dining"].length - 2]) < 0 
                          ? <span className="text-warning mr-2"> <i className="fas fa-arrow-down" />
                          {Math.abs(((Object.values(rest["performance_dining"][rest["performance_dining"].length - 1]) 
                          -Object.values(rest["performance_dining"][rest["performance_dining"].length - 2])) / Object.values(rest["performance_dining"][rest["performance_dining"].length - 2])) * 100).toFixed(2)}%
                          </span>: 
                          <span className="text-success mr-2">
                          <i className="fas fa-arrow-up" /> {Math.abs(((Object.values(rest["performance_dining"][rest["performance_dining"].length - 1]) 
                          -Object.values(rest["performance_dining"][rest["performance_dining"].length - 2])) / Object.values(rest["performance_dining"][rest["performance_dining"].length - 2])) * 100).toFixed(2)}%
                          </span>
                        }
                        {" "}
                        <span className="text-nowrap">Since last 2 weeks</span>
                      </p>
                      )
                        :
                          "NA"
                    }
                  </CardBody>
                </Card>
              </Col>
              <Col lg="6" xl="3">
                <Card className="card-stats mb-4 mb-xl-0">
                  <CardBody>
                    <Row>
                      <div className="col">
                        <CardTitle
                          tag="h5"
                          className="text-uppercase text-muted mb-0"
                        >
                          Delivery Performance 
                        </CardTitle>
                        <span className="h2 font-weight-bold mb-0">{
                       rest["dining or delivery"] === "Dining and Delivery" || rest["dining or delivery"] === "Delivery"
                        ?
                          Object.values(rest["performance_delivery"][rest["performance_delivery"].length - 1])
                        :
                          "NA"
                        }
                        </span>
                      </div>
                      <Col className="col-auto">
                        <div className="icon icon-shape bg-info text-white rounded-circle shadow">
                          <i className="fas fa-percent" />
                        </div>
                      </Col>
                    </Row>
                    {
                       rest["dining or delivery"] === "Dining and Delivery" || rest["dining or delivery"] === "Delivery"
                        ?
                        (<p className="mt-3 mb-0 text-muted text-sm">
                          {Object.values(rest["performance_delivery"][rest["performance_delivery"].length - 1]) 
                          -Object.values(rest["performance_delivery"][rest["performance_delivery"].length - 2]) < 0 
                          ? <span className="text-warning mr-2"> <i className="fas fa-arrow-down" />
                          {Math.abs(((Object.values(rest["performance_delivery"][rest["performance_delivery"].length - 1]) 
                          -Object.values(rest["performance_delivery"][rest["performance_delivery"].length - 2])) / Object.values(rest["performance_delivery"][rest["performance_delivery"].length - 2])) * 100).toFixed(2)}%
                          </span>: 
                          <span className="text-success mr-2">
                          <i className="fas fa-arrow-up" /> {Math.abs(((Object.values(rest["performance_delivery"][rest["performance_delivery"].length - 1]) 
                          -Object.values(rest["performance_delivery"][rest["performance_delivery"].length - 2])) / Object.values(rest["performance_delivery"][rest["performance_delivery"].length - 2])) * 100).toFixed(2)}%
                          </span>
                        }
                        {" "}
                        <span className="text-nowrap">Since last 2 weeks</span>
                      </p>
                      )
                        :
                          "NA"
                    }
                  </CardBody>
                </Card>
              </Col>
            </Row>
          </div>
        </Container>
      </div>
    </>
  );
};

export default Header;
