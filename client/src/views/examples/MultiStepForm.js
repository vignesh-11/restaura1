import React, { useState, useRef } from "react";
import {
  Typography,
  TextField,
  Checkbox,
  Button,
  Stepper,
  Step,
  StepLabel,
} from "@material-ui/core";
import { makeStyles } from "@material-ui/core/styles";
import { CssBaseline, Container, Paper, Box, MenuItem, Select, OutlinedInput, FormControl, InputLabel} from "@material-ui/core";
import {
  useForm,
  Controller,
  FormProvider,
  useFormContext,

} from "react-hook-form";

import FormControlLabel from '@material-ui/core/FormControlLabel';

import PieChart from "variables/pieChart";
import BarChart from "variables/barChart";
import LineChart from "variables/lineChart";
import DualBarChart from "variables/dualBarChart";

const useStyles = makeStyles((theme) => ({
  button: {
    marginRight: theme.spacing(1),
  },
}));

function getSteps() {
  return [
    "Instructions",
    "Online Conformity",
    "Restaurant Type & Cuisines offered",
    "Cost",
    "Location",
  ];
}
const BasicForm = () => {
  const { control } = useFormContext();
  return (
    <>
      <br />
      <p>1. This form is used to take input regarding your restaurant like whether your restaurant takes online orderes,
        location details, restaurant type, cuisines offered etc.
      </p>
      <br />
      <p>
        2. These details will then be analysed and report will be generated predicting the success of your restaurant.
      </p>
      <br />
      <p>
        3. All the input fields are to be filled in compulsorily for accurate result.
      </p>
      <br />
      <p>
        4. These results will then help you in your decisions regarding opening of the restaurant.
      </p>
      <br />
      <p>
        5. In order to assist you with filling the form, at each step we show you the current trends and analysis.
      </p>
      <Controller
        control={control}
        name="name"
        render={({ field }) => (
          <TextField
            required
            id="nick-name"
            label="Restaurant Name"
            variant="outlined"
            placeholder="Enter Your Restaurant Name"
            fullWidth
            margin="normal"
            {...field}
          />
        )}
      />
    </>
  );
};
const OnlineConfForm = () => {
  const { control } = useFormContext();
  return (
    <>
      <Controller
        control={control}
        name="online_order"
        render={({ field }) => (
          <FormControlLabel
                control={<Checkbox {...field} />}
                label='Online order'
            />
        )}
      />
      <br />
      <br />
      <Controller
        control={control}
        name="book_table"
        render={({ field }) => (
          <FormControlLabel
                control={<Checkbox {...field} />}
                label='Online booking table'
            />
        )}
      />
      <br />
      <br />
    </>
  );
};


const TypeCuisForm = () => {

    const restTypesNames = ["Bakery", "Bar", "Beverage Shop", "Bhojanalya", "Cafe", "Casual Dining", "Club", "Confectionery", "Delivery"
                        , "Dessert Parlor", "Dhaba", "Fine Dining", "Food Court", "Food Truck", "Irani Cafee", "Kiosk",
                        "Lounge", "Meat Shop", "Mess", "Microbrewery", "Pop Up", "Pub", "Quick Bites", "Sweet Shop", "Takeaway"];

    const cuisines = [
                    "Afghan",
                    "Afghani",
                    "African",
                    "American",
                    "Andhra",
                    "Arabian",
                    "Asian",
                    "Assamese",
                    "Australian",
                    "Awadhi",
                    "BBQ",
                    "Bakery",
                    "Bar Food",
                    "Belgian",
                    "Bengali",
                    "Beverages",
                    "Bihari",
                    "Biryani",
                    "Bohri",
                    "British",
                    "Bubble Tea",
                    "Burger",
                    "Burmese",
                    "Cafe",
                    "Cantonese",
                    "Charcoal Chicken",
                    "Chettinad",
                    "Chinese",
                    "Coffee",
                    "Continental",
                    "Desserts",
                    "Drinks Only",
                    "European",
                    "Fast Food",
                    "Finger Food",
                    "French",
                    "German",
                    "Goan",
                    "Greek",
                    "Grill",
                    "Gujarati",
                    "Healthy Food",
                    "Hot dogs",
                    "Hyderabadi",
                    "Ice Cream",
                    "Indian",
                    "Indonesian",
                    "Iranian",
                    "Italian",
                    "Japanese",
                    "Jewish",
                    "Juices",
                    "Kashmiri",
                    "Kebab",
                    "Kerala",
                    "Konkan",
                    "Korean",
                    "Lebanese",
                    "Lucknowi",
                    "Maharashtrian",
                    "Malaysian",
                    "Malwani",
                    "Mangalorean",
                    "Mediterranean",
                    "Mexican",
                    "Middle Eastern",
                    "Mithai",
                    "Modern Indian",
                    "Momos",
                    "Mongolian",
                    "Mughlai",
                    "Naga",
                    "Nepalese",
                    "North Eastern",
                    "North Indian",
                    "Oriya",
                    "Paan",
                    "Pan Asian",
                    "Parsi",
                    "Pizza",
                    "Portuguese",
                    "Rajasthani",
                    "Raw Meats",
                    "Roast Chicken",
                    "Rolls",
                    "Russian",
                    "Salad",
                    "Sandwich",
                    "Seafood",
                    "Sindhi",
                    "Singaporean",
                    "South American",
                    "South Indian",
                    "Spanish",
                    "Sri Lankan",
                    "Steak",
                    "Street Food",
                    "Sushi",
                    "Tamil",
                    "Tea",
                    "Tex-Mex",
                    "Thai",
                    "Tibetan",
                    "Turkish",
                    "Vegan",
                    "Vietnamese",
                    "Wraps"    
                  ];


  const { control } = useFormContext();
  const [cuis, setCuis] = React.useState([]);
  const handleChange1 = (event) => {
    const {
      target: { value },
    } = event;
    setCuis(
      // On autofill we get a the stringified value.
      typeof value === 'string' ? value.split(',') : value,
    );
  };

  const [restType, setRestType] = React.useState([]);
  const handleChange2 = (event) => {
    const {
      target: { value },
    } = event;
    setRestType(
      // On autofill we get a the stringified value.
      typeof value === 'string' ? value.split(',') : value,
    );
  };

  return (
    <>
    <FormControl>
      Select restaurant type
      <Controller
            control={control}
            name="rest_type"
            render={({ field }) => {
              return (
                <Select
                  required
                  multiple
                  value={restType}
                  onChange={handleChange2}
                  input={<OutlinedInput label="Name" />}
                  {...field}
                >
                  {restTypesNames.map((name) => (
                    <MenuItem
                      key={name}
                      value={name}
                    >
                      {name}
                    </MenuItem>
                  ))}
                </Select>
              );
            }}
          />
      </FormControl>
      <br />
      <br />
      <br />


      <FormControl>
      Select the cuisines you wish to serve
      <Controller
            control={control}
            name="cuisines"
            render={({ field }) => {
              return (
                <Select
                  required
                  multiple
                  value={cuis}
                  onChange={handleChange1}
                  input={<OutlinedInput label="Name" />}
                  {...field}
                >
                  {cuisines.map((name) => (
                    <MenuItem
                      key={name}
                      value={name}
                    >
                      {name}
                    </MenuItem>
                  ))}
                </Select>
              );
            }}
          />
      </FormControl>
      <br />
      <br />
      <br />


    </>
  );
};
const CostForm = () => {
  const { control } = useFormContext();
  return (
    <>
      <Controller
        control={control}
        name="approx_cost(for two people)"
        render={({ field }) => (
          <TextField
            required
            id="cost"
            label="Cost for 2 persons in Rs."
            variant="outlined"
            type="number"
            InputProps={{ inputProps: { min: 0, max: 10000 } }}
            placeholder="Cost"
            fullWidth
            margin="normal"
            {...field}
          />
        )}
      />
      <br />
      <br />
    </>
  );
};

const LocationForm = () => {

  const { control } = useFormContext();
  const [loc, setLoc] = React.useState([]);
  const handleChange = (event) => {
    setLoc(event.target.value);
  };

  return (
    <>
    <FormControl>
      Select Restaurant Location
      <Controller
            control={control}
            name="location"
            render={({ field }) => {
              return (
                <Select
                  required
                  value={loc}
                  onChange={handleChange}
                  autoWidth
                  label="Location"
                  {...field}
                >
                      <MenuItem value="BTM">BTM</MenuItem>
                      <MenuItem value="Banashankari">Banashankari</MenuItem>
                      <MenuItem value="Banaswadi">Banaswadi</MenuItem>
                      <MenuItem value="Bannerghatta Road">Bannerghatta Road</MenuItem>
                      <MenuItem value="Basavanagudi">Basavanagudi</MenuItem>
                      <MenuItem value="Basaveshwara Nagar">Basaveshwara Nagar</MenuItem>
                      <MenuItem value="Bellandur">Bellandur</MenuItem>
                      <MenuItem value="Bommanahalli">Bommanahalli</MenuItem>
                      <MenuItem value="Brigade Road">Brigade Road</MenuItem>
                      <MenuItem value="Brookefield">Brookefield</MenuItem>
                      <MenuItem value="CV Raman Nagar">CV Raman Nagar</MenuItem>
                      <MenuItem value="Central Bangalore">Central Bangalore</MenuItem>
                      <MenuItem value="Church Street">Church Street</MenuItem>
                      <MenuItem value="City Market">City Market</MenuItem>
                      <MenuItem value="Commercial Street">Commercial Street</MenuItem>
                      <MenuItem value="Cunningham Road">Cunningham Road</MenuItem>
                      <MenuItem value="Domlur">Domlur</MenuItem>
                      <MenuItem value="East Bangalore">East Bangalore</MenuItem>
                      <MenuItem value="Ejipura">Ejipura</MenuItem>
                      <MenuItem value="Electronic City">Electronic City</MenuItem>
                      <MenuItem value="Frazer Town">Frazer Town</MenuItem>
                      <MenuItem value="HBR Layout">HBR Layout</MenuItem>
                      <MenuItem value="HSR">HSR</MenuItem>
                      <MenuItem value="Hebbal">Hebbal</MenuItem>
                      <MenuItem value="Hennur">Hennur</MenuItem>
                      <MenuItem value="Hosur Road">Hosur Road</MenuItem>
                      <MenuItem value="ITPL Main Road, Whitefield">ITPL Main Road, Whitefield</MenuItem>
                      <MenuItem value="Indiranagar">Indiranagar</MenuItem>
                      <MenuItem value="Infantry Road">Infantry Road</MenuItem>
                      <MenuItem value="JP Nagar">JP Nagar</MenuItem>
                      <MenuItem value="Jakkur">Jakkur</MenuItem>
                      <MenuItem value="Jalahalli">Jalahalli</MenuItem>
                      <MenuItem value="Jayanagar">Jayanagar</MenuItem>
                      <MenuItem value="Jeevan Bhima Nagar">Jeevan Bhima Nagar</MenuItem>
                      <MenuItem value="KR Puram">KR Puram</MenuItem>
                      <MenuItem value="Kaggadasapura">Kaggadasapura</MenuItem>
                      <MenuItem value="Kalyan Nagar">Kalyan Nagar</MenuItem>
                      <MenuItem value="Kammanahalli">Kammanahalli</MenuItem>
                      <MenuItem value="Kanakapura Road">Kanakapura Road</MenuItem>
                      <MenuItem value="Kengeri">Kengeri</MenuItem>
                      <MenuItem value="Koramangala 1st Block">Koramangala 1st Block</MenuItem>
                      <MenuItem value="Koramangala 2nd Block">Koramangala 2nd Block</MenuItem>
                      <MenuItem value="Koramangala 3rd Block">Koramangala 3rd Block</MenuItem>
                      <MenuItem value="Koramangala 4th Block">Koramangala 4th Block</MenuItem>
                      <MenuItem value="Koramangala 5th Block">Koramangala 5th Block</MenuItem>
                      <MenuItem value="Koramangala 6th Block">Koramangala 6th Block</MenuItem>
                      <MenuItem value="Koramangala 7th Block">Koramangala 7th Block</MenuItem>
                      <MenuItem value="Koramangala 8th Block">Koramangala 8th Block</MenuItem>
                      <MenuItem value="Koramangala">Koramangala</MenuItem>
                      <MenuItem value="Kumaraswamy Layout">Kumaraswamy Layout</MenuItem>
                      <MenuItem value="Langford Town">Langford Town</MenuItem>
                      <MenuItem value="Lavelle Road">Lavelle Road</MenuItem>
                      <MenuItem value="MG Road">MG Road</MenuItem>
                      <MenuItem value="Magadi Road">Magadi Road</MenuItem>
                      <MenuItem value="Majestic">Majestic</MenuItem>
                      <MenuItem value="Malleshwaram">Malleshwaram</MenuItem>
                      <MenuItem value="Marathahalli">Marathahalli</MenuItem>
                      <MenuItem value="Mysore Road">Mysore Road</MenuItem>
                      <MenuItem value="Nagarbhavi">Nagarbhavi</MenuItem>
                      <MenuItem value="Nagawara">Nagawara</MenuItem>
                      <MenuItem value="New BEL Road">New BEL Road</MenuItem>
                      <MenuItem value="North Bangalore">North Bangalore</MenuItem>
                      <MenuItem value="Old Airport Road">Old Airport Road</MenuItem>
                      <MenuItem value="Old Madras Road">Old Madras Road</MenuItem>
                      <MenuItem value="Peenya">Peenya</MenuItem>
                      <MenuItem value="RT Nagar">RT Nagar</MenuItem>
                      <MenuItem value="Race Course Road">Race Course Road</MenuItem>
                      <MenuItem value="Rajajinagar">Rajajinagar</MenuItem>
                      <MenuItem value="Rajarajeshwari Nagar">Rajarajeshwari Nagar</MenuItem>
                      <MenuItem value="Rammurthy Nagar">Rammurthy Nagar</MenuItem>
                      <MenuItem value="Residency Road">Residency Road</MenuItem>
                      <MenuItem value="Richmond Road">Richmond Road</MenuItem>
                      <MenuItem value="Sadashiv Nagar">Sadashiv Nagar</MenuItem>
                      <MenuItem value="Sahakara Nagar">Sahakara Nagar</MenuItem>
                      <MenuItem value="Sanjay Nagar">Sanjay Nagar</MenuItem>
                      <MenuItem value="Sankey Road">Sankey Road</MenuItem>
                      <MenuItem value="Sarjapur Road">Sarjapur Road</MenuItem>
                      <MenuItem value="Seshadripuram">Seshadripuram</MenuItem>
                      <MenuItem value="Shanti Nagar">Shanti Nagar</MenuItem>
                      <MenuItem value="Shivajinagar">Shivajinagar</MenuItem>
                      <MenuItem value="South Bangalore">South Bangalore</MenuItem>
                      <MenuItem value="St. Marks Road">St. Marks Road</MenuItem>
                      <MenuItem value="Thippasandra">Thippasandra</MenuItem>
                      <MenuItem value="Ulsoor">Ulsoor</MenuItem>
                      <MenuItem value="Uttarahalli">Uttarahalli</MenuItem>
                      <MenuItem value="Varthur Main Road, Whitefield">Varthur Main Road, Whitefield</MenuItem>
                      <MenuItem value="Vasanth Nagar">Vasanth Nagar</MenuItem>
                      <MenuItem value="Vijay Nagar">Vijay Nagar</MenuItem>
                      <MenuItem value="West Bangalore">West Bangalore</MenuItem>
                      <MenuItem value="Whitefield">Whitefield</MenuItem>
                      <MenuItem value="Wilson Garden">Wilson Garden</MenuItem>
                      <MenuItem value="Yelahanka">Yelahanka</MenuItem>
                      <MenuItem value="Yeshwantpur">Yeshwantpur</MenuItem>
              </Select>
              );
            }}
          />
      </FormControl>
      <br />
      <br />
    </>
  );

};

function getStepContent(step) {
  switch (step) {
    case 0:
      return(
        <>
         <BasicForm />
        </>
      );

    case 1:
      return(
        <>
         <OnlineConfForm />
         <PieChart />
         <br />
         <br />
        </>
      );
    case 2:
      return (
        <>
          <TypeCuisForm />
          <DualBarChart />
          <br />
          <br />
        </>
      )
    case 3:
      return (
          <>
             <CostForm />
             <LineChart />
             <br />
             <br />
          </>
      );
    case 4:
      return(
        <>
          <LocationForm />
          <BarChart />
          <br />
          <br />
        </>
      );
    default:
      return "unknown step";
  }
}

const LinearStepper = () => {
  const classes = useStyles();
  var finalData = useRef({});
  const prob = useRef({});
  const methods = useForm({
    defaultValues: {
      "url": "None",
      "address": "None",
      "name": "",
      "online_order": false,
      "book_table": false,
      "rate": "None",
      "votes": "0",
      "phone": "123142424",
      "dish_liked": "None",
      "reviews_list": "None",
      "menu_item": "[]",
      "location": "",
      "rest_type": [],
      "listed_in(type)": [],
      "listed_in(city)": [],
      "cuisines": [],
      "approx_cost(for two people)": "",
    },
  });
  const [activeStep, setActiveStep] = useState(0);
  const [skippedSteps, setSkippedSteps] = useState([]);
  const [loader, setLoader] = useState(true);
  const steps = getSteps();

  const isStepOptional = (step) => {
    return step === 1 || step === 2;
  };

  const isStepSkipped = (step) => {
    return skippedSteps.includes(step);
  };

  const handleFinal = (data) => {
      //console.log(data);
      
      data["listed_in(type)"] = data["rest_type"][0];
      let restType = "";
      let i;
      for(i=0; i<data["rest_type"].length - 1; i++){
        restType +=  data["rest_type"][i] + ", ";
      }
      restType +=  data["rest_type"][i];
      data["rest_type"] = restType;

      let cui = "";
      for(i=0; i<data["cuisines"].length - 1; i++){
        cui +=  data["cuisines"][i] + ", ";
      }
      cui +=  data["cuisines"][i];
      data["cuisines"] = cui;


      data["listed_in(city)"] = data["location"];
      data["book_table"] ? data["book_table"] = "Yes" : data["book_table"] = "No";
      data["online_order"] ? data["online_order"] = "Yes" : data["online_order"] = "No";

      finalData.current = data;

      fetch("http://127.0.0.1:5000/predict", {
        method: 'post',
        headers: {
          'Accept': 'application/json, text/plain, */*',
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
      })
        .then((data) => data.json())
        .then((res) => {
          var reports = JSON.parse(localStorage.getItem("reports") || "[]");
          data.id =  Math.floor(Math.random() * 1000000)
          data.success_prob = res.success_prob;
          reports.push(data);
          localStorage.setItem("reports", JSON.stringify(reports));
          prob.current = res.success_prob;
          setLoader(false);
        });
  }

  const handleDownload = (data) => {
    fetch("http://127.0.0.1:5000/report", {
      method: 'post',
      headers: {
        'Accept': 'application/json, text/plain, */*',
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(data)
    })
    .then(response => response.blob())
    .then(blob => URL.createObjectURL(blob))
    .then(url => {
        window.open(url, '_blank');
        URL.revokeObjectURL(url);
    });
  }

  const handleNext = (data) => {
    console.log(data);
    if (activeStep === steps.length - 1) {
      finalData.current = {...data};
      //console.log(finalData);
      setActiveStep(activeStep + 1);
    } else {
      setActiveStep(activeStep + 1);
      setSkippedSteps(
        skippedSteps.filter((skipItem) => skipItem !== activeStep)
      );
    }
  };

  const handleBack = () => {
    setActiveStep(activeStep - 1);
  };

  const handleSkip = () => {
    if (!isStepSkipped(activeStep)) {
      setSkippedSteps([...skippedSteps, activeStep]);
    }
    setActiveStep(activeStep + 1);
  };

  // const onSubmit = (data) => {
  //   console.log(data);
  // };
  return (
    <div>
      <Stepper alternativeLabel activeStep={activeStep}>
        {steps.map((step, index) => {
          const labelProps = {};
          const stepProps = {};
          if (isStepSkipped(index)) {
            stepProps.completed = false;
          }
          return (
            <Step {...stepProps} key={index}>
              <StepLabel {...labelProps}>{step}</StepLabel>
            </Step>
          );
        })}
      </Stepper>
      {activeStep===steps.length && loader ?  
      <Typography variant="h3" align="center">
          Loading.....
      </Typography> : null}
      {activeStep===steps.length && loader ? handleFinal(finalData.current) : null}
      {activeStep===steps.length && !loader ? 
      <div>
        <Typography variant="h4" align="center">
            Success percentage : {prob.current.toFixed(2)} %
        </Typography> 
        <br />
        <Typography align="center">
          Download report <span style={{"cursor": "pointer", "textDecoration":"underline", "backgroundColor":"#acacdc"}} onClick={() => handleDownload(finalData.current)}>here</span>
        </Typography>
      </div>
      : null}
      {activeStep === steps.length ? null : (
        <>
          <FormProvider {...methods}>
            <form onSubmit={methods.handleSubmit(handleNext)}>
              {getStepContent(activeStep)}

              <Button
                className={classes.button}
                disabled={activeStep === 0}
                onClick={handleBack}
              >
                back
              </Button>
              <Button
                className={classes.button}
                variant="contained"
                color="primary"
                // onClick={handleNext}
                type="submit"
              >
                {activeStep === steps.length - 1 ? "Finish" : "Next"}
              </Button>
            </form>
          </FormProvider>
        </>
      )}
    </div>
  );
};

const MultiStepForm = () => {
    return (
        <>
          <CssBaseline />
          <Container component={Box} p={4}>
            <Paper component={Box} p={3}>
              <LinearStepper />
            </Paper>
          </Container>
        </>
      );
}

export default MultiStepForm;