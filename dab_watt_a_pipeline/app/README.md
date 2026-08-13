# An app for visuals

Decided to throw one of these in just to make it easier to see the data.

It's a one shot ai prompt so don't judge but if you can get it deployed it should help visualise the results. Just make sure you have deployed to the prod target, I didn't want to faff with a dev version of the app.

You'll need to go and manually start the app and point at the app folder in the /Workspace/Bundles/dab_watt_a_pipeline/prod/files
/app  deploy location. I've automated this stuff before just didn't want to over complicate it in the time I have.

And don't forget to grant the apps SP access to the tables, would have used OBO but again way too much for such a small demo.

The App should pick up table locations from the bundle deploy but if it's wrong check the defaults in the resources/app.yml

#TODO add in an easter egg