import {
  Button,
  Card,
  CardHeader,
  CardBody,
  FormGroup,
  Form,
  Input,
  InputGroupAddon,
  InputGroupText,
  InputGroup,
  Row,
  Col,
} from "reactstrap";
import { Link } from "react-router-dom";

const Landing = () => {
  return (
    <>
      <Col lg="7" md="9">
        <Card className="bg-secondary shadow border-0">
          <CardHeader className="bg-transparent pb-5">
            <div className="text-muted text-center mt-2 mb-3">
              <small style={{ fontSize: "2rem" }}>Please select</small>
            </div>
          </CardHeader>
          <CardBody className="px-lg-5 py-lg-5">
          <Row className="mt-3">
          <Col xs="6">
            <Link
              className="text-dark"
              to="/business/open"
              tag={Link}
            >
              <div className="text-center">
                <i className="fas fa-building fa-8x"></i>
                <br />
                <br />
                <small>Want to start your own restaurant?</small>
              </div>
            </Link>
          </Col>
          <Col className="text-right" xs="6">
            <Link
              className="text-dark"
              to="/customer/leaderboard"
              tag={Link}
            >
              <div className="text-center">
                <i className="fas fa-pizza-slice fa-8x"></i>
                <br />
                <br />
                <small>Want to find a good place to eat?</small>
              </div>
            </Link>
          </Col>
        </Row>
          </CardBody>
        </Card>
      </Col>
    </>
  );
};

export default Landing;
